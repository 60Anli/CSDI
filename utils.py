import numpy as np
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    resume_path=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
        checkpoint_last_path = foldername + "/checkpoint_last.pth"
        checkpoint_best_path = foldername + "/checkpoint_best.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    start_epoch = 0
    best_valid_loss = 1e10
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=next(model.parameters()).device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", -1) + 1
            best_valid_loss = checkpoint.get("best_valid_loss", best_valid_loss)
            print("resumed training from", resume_path, "at epoch", start_epoch)
        else:
            model.load_state_dict(checkpoint)
            print("loaded model weights from", resume_path)

    if getattr(model, "raap", None) is not None:
        bank_count = model.build_raap_bank(train_loader)
        print(f"RAAP retrieval bank built with {bank_count} items")
    raap_bank_rebuild_interval = getattr(model, "raap_bank_rebuild_epoch_interval", 0)

    def save_checkpoint(path, epoch_no):
        torch.save(
            {
                "epoch": epoch_no,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "best_valid_loss": best_valid_loss,
                "config": config,
            },
            path,
        )

    for epoch_no in range(start_epoch, config["epochs"]):
        if (
            getattr(model, "raap", None) is not None
            and raap_bank_rebuild_interval > 0
            and epoch_no > start_epoch
            and epoch_no % raap_bank_rebuild_interval == 0
        ):
            bank_count = model.build_raap_bank(train_loader)
            print(f"RAAP retrieval bank rebuilt with {bank_count} items at epoch {epoch_no}")

        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                if foldername != "":
                    torch.save(model.state_dict(), foldername + "/tmp_model" + str(epoch_no) + ".pth")
                    save_checkpoint(checkpoint_best_path, epoch_no)
        if foldername != "":
            save_checkpoint(checkpoint_last_path, epoch_no)

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        prior_mse_total = 0
        prior_mae_total = 0
        prior_evalpoints_total = 0
        has_raap_prior = False
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                if len(output) == 6:
                    samples, c_target, eval_points, observed_points, observed_time, raap_prior = output
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                    raap_prior = None
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                if raap_prior is not None:
                    raap_prior = raap_prior.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                postfix = {
                    "rmse_total": np.sqrt(mse_total / evalpoints_total),
                    "mae_total": mae_total / evalpoints_total,
                    "batch_no": batch_no,
                }
                if raap_prior is not None:
                    has_raap_prior = True
                    prior_mse_current = (
                        ((raap_prior - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    prior_mae_current = (
                        torch.abs((raap_prior - c_target) * eval_points)
                    ) * scaler
                    prior_mse_total += prior_mse_current.sum().item()
                    prior_mae_total += prior_mae_current.sum().item()
                    prior_evalpoints_total += eval_points.sum().item()
                    postfix["raap_prior_mae"] = prior_mae_total / prior_evalpoints_total

                it.set_postfix(ordered_dict=postfix, refresh=True)

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)

            if has_raap_prior:
                prior_rmse = np.sqrt(prior_mse_total / prior_evalpoints_total)
                prior_mae = prior_mae_total / prior_evalpoints_total
                with open(
                    foldername + "/raap_prior_result_nsample" + str(nsample) + ".pk",
                    "wb",
                ) as f:
                    pickle.dump([prior_rmse, prior_mae], f)
                print("RAAP_prior_RMSE:", prior_rmse)
                print("RAAP_prior_MAE:", prior_mae)

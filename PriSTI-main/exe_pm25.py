from exe_aqi36 import build_parser, main


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "PriSTI PM25"
    args = parser.parse_args()
    print(args)
    main(args)

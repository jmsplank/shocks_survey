from qpara import correct_win_dict, main

if __name__ == "__main__":
    trange = ["2020-03-18/02:47:24", "2020-03-18/03:08:59"]
    win_dict = {
        "ms": ["2020-03-18 02:47:24.8036", "2020-03-18 03:02:29.9562"],
        "fs": ["2020-03-18 03:02:31.5808", "2020-03-18 03:08:59.7503"],
    }
    win_dict = correct_win_dict(win_dict)
    print(win_dict)
    main(trange, win_dict)

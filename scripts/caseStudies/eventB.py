from qpara import correct_win_dict, main

if __name__ == "__main__":
    trange = ["2018-03-16/01:39:59", "2018-03-16/01:59:17"]
    win_dict = {
        "ms": ["2018-03-16 01:39:59.2322", "2018-03-16 01:50:46.8236"],
        "shock": ["2018-03-16 01:50:42.9398", "2018-03-16 01:54:27.4913"],
        "sw": ["2018-03-16 01:54:19.008", "2018-03-16 01:59:17.0472"],
    }
    win_dict = correct_win_dict(win_dict)
    print(win_dict)
    main(trange, win_dict)

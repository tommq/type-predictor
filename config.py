class Config:
    dispatcher_threshold = 90
    dispatcher_min_interval = 8000
    dispatcher_window_size_before = 30
    dispatcher_window_size_after = 120
    dispatcher_step_size = 1
    dispatcher_persistence = True
    # Number of worker processes
    workers = 4
    accepted_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                           's', 't', 'u', 'v', 'w', 'y', 'x', 'z', ' ']
    # Output options
    dict_sep_threshold = 5

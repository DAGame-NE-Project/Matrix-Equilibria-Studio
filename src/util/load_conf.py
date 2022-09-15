def update_args(args, dic):
    for x,y in dic.items():
        setattr(args, x, y)

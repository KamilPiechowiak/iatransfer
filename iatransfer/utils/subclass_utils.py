def get_subclasses(clazz: type):
    subclasses = {}
    classes = [clazz]
    while len(classes) > 0:
        clazz = classes.pop()
        for c in clazz.__subclasses__():
            subclasses[c.__name__] = c
            classes.append(c)
    return subclasses

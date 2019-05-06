from functools import partial
def greeting(my_greeting, name):
    print("%s %s" % (my_greeting, name))

# uses partial function to pass hello to the my_greeting variable
say_hello_to = partial(greeting, "Hello")
say_hello_to("World")
say_hello_to("Dog")
say_hello_to("Cat")
import sys
from ImageManager import ImageManager


def close_program():
    print("Bye!")
    sys.exit(0)


class Switcher:

    options = None
    image_manager: ImageManager = None

    def __init__(self):
        self.image_manager = ImageManager()
        self.options = {
            1: self.load_image,
            2: self.process_image,
            3: self.unload_image,
            4: close_program
        }

    def process_option(self, option):
        func = self.options.get(option, lambda: print("Invalid option. Try again."))
        func()

    def load_image(self):
        filename = input("Please specify the image name to load: ")
        self.image_manager.def_set_image()
#        self.image_manager.set_image(filename)

    def process_image(self):
        self.image_manager.process_image()

    def unload_image(self):
        self.image_manager.unload_image()

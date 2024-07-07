from Switcher import Switcher


# Menu of the program
def print_menu():
    print("Hip X-Ray image processing")
    print("Options: ")
    print("1. Load image")
    print("2. Process image")
    print("3. Unload image")
    print("4. Close")

    option = input("Please choose one of the options above: ")
    return option


# Main block, used to interact with the user. Select option 4 to exit
if __name__ == '__main__':
    switcher = Switcher()
    menu_option = -1

    while menu_option != 4:
        menu_option = int(print_menu())

        switcher.process_option(menu_option)

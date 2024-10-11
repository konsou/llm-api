import colorama


def print_yellow(text: str):
    print(f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}")


print_warning = print_yellow

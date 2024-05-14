from accelerate import PartialState


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def is_main_process() -> bool:
    return PartialState().is_main_process


def get_local_rank() -> int:
    """The index of the current process on the current server."""
    return PartialState().local_process_index


def num_processes() -> int:
    return PartialState().num_processes


def wait_for_everyone() -> None:
    PartialState().wait_for_everyone()


def print_on_rank_0(*args, **kwargs) -> None:
    if not is_main_process():
        return

    print(*args, **kwargs)


def print_warning_on_rank_0(message: str, **kwargs) -> None:
    print_on_rank_0(f'{bcolors.WARNING}Warning: {message}{bcolors.ENDC}', **kwargs)

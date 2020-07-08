#!/usr/bin/env python3
"""Utility functions for pretty-printing to console."""
import os

# extract dimensions of current terminal window
TERM_DIM = os.popen('stty size', 'r').read().split()


def print_header(header_string):
    """Print underline-formatted title text to console.

    Args:
        header_string (str): string of header/title text to be printed
    """
    print('\n')  # + '-'*int(TERM_DIM[1]))
    print(header_string)
    print('-' * len(header_string))


def print_div():
    """Print divider to console (two console-width dashed lines)."""
    print('\n')
    print('-' * int(TERM_DIM[1]))
    print('-' * int(TERM_DIM[1]))

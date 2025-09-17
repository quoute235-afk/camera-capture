"""Entry point for the dual invert camera application."""

import tkinter as tk

from gui import DualInvertApp


def main() -> None:
    root = tk.Tk()
    DualInvertApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

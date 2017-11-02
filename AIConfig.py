import threading
from tkinter import Tk, Label, Entry, StringVar, IntVar


class AIConfig(threading.Thread):

    def __init__(self, algorithm):
        threading.Thread.__init__(self)
        self.root = None
        self.algorithm = algorithm

    def entry_on_change(self, sv, i):
        try:
            print(sv, self.epsilon_set_value)
            if sv == self.epsilon_set_value:
                self.algorithm.epsilon = int(self.epsilon_set_value.get())
        except Exception as e:
            print(e)

    def _epsilon(self):
        self.epsilon_get_value = StringVar()
        self.epsilon_set_value = StringVar()
        self.epsilon_set_value.trace("w", lambda name, index, mode, var=self.epsilon_set_value, i=0: self.entry_on_change(var, i))
        self.epsilon_label = Label(text="Epsilon:")
        self.epsilon_label.place(x=25, y=25)

        self.epsilon_input = Entry(textvariable=self.epsilon_set_value)
        self.epsilon_input.place(x=70, y=25, width=50)

        self.epsilon_label = Label(textvariable=self.epsilon_get_value)
        self.epsilon_label.place(x=100, y=25)

    def construct(self):
        self._epsilon()

    def update_loop(self):
        self.epsilon_get_value.set(self.algorithm.epsilon)

        self.root.after(500, self.update_loop)

    def run(self):
        self.root = Tk()
        self.root.resizable(width=False, height=False)
        self.root.geometry('{}x{}'.format(300, 600))
        self.root.title("DeepLineWars TODO get from config")
        self.construct()
        self.update_loop()
        self.root.mainloop()



from tkinter import *

from tkinter import messagebox
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''import tkinter as tk
    from tkinter import ttk

    # Creating tkinter window and set dimensions
    window = tk.Tk()
    window.title('Combobox')
    window.geometry('500x250')

    # label text for title
    ttk.Label(window, text="Choose the country and vote for them",
              background='cyan', foreground="black",
              font=("Times New Roman", 15)).grid(row=0, column=1)

    # Set label
    ttk.Label(window, text="Select the Country :",
              font=("Times New Roman", 12)).grid(column=0,
                                                 row=5, padx=5, pady=25)
    ttk.Label(window, text="Select the Country :",
              font=("Times New Roman", 12)).grid(column=0,
                                                 row=6, padx=5, pady=25)
    # Create Combobox
    n = tk.StringVar()
    country = ttk.Combobox(window, width=27, textvariable=n)

    # Adding combobox drop down list
    country['values'] = (' India',
                         ' China',
                         ' Australia',
                         ' Nigeria',
                         ' Malaysia',
                         ' Italy',
                         ' Turkey',
                         ' Canada')

    country.grid(column=1, row=6)
    country.current()

    n = tk.StringVar()
    country = ttk.Combobox(window, width=27, textvariable=n)

    # Adding combobox drop down list
    country['values'] = (' India',
                         ' China',
                         ' Australia',
                         ' Nigeria',
                         ' Malaysia',
                         ' Italy',
                         ' Turkey',
                         ' Canada')

    country.grid(column=1, row=5)
    country.current()

    window.mainloop()




'''


    '''from tkinter import *
    from tkinter import ttk

    z=[]
    def retrieve():
        z.append(Combo.get())
        print(z)
        print(Combo.get())


    root = Tk()
    root.geometry("400x250")

    frame = Frame(root)
    frame.pack()

    vlist = ["Option1", "Option2", "Option3",
             "Option4", "Option5"]
    var = StringVar()
    label = Label(root, textvariable=var, relief=RAISED,pady=10,padx=10)
    var.set("Hey!? How are you doing?")
    label.place(relx=0., rely=0.15, anchor="sw")
    Combo = ttk.Combobox(frame, values=vlist)
    Combo.set("Pick an Option")
    Combo.pack(padx=20, pady=20,relx="after",rely=0.15)

    Button = Button(frame, text="Submit", command=retrieve)
    print()
    Button.pack(padx=25, pady=25)

    root.mainloop()'''
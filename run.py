#import module from tkinter for UI
from tkinter import *
import os
from datetime import datetime;
#creating instance of TK
root=Tk()

root.configure(background="white")

#root.geometry("300x300")

def function1():
    
    os.system("python snap.py")
     
def function2():

    os.system("python test.py")

def function3():
    os.system("python reg.py")


def function4():
    os.system("python reset.py")

#stting title for the window
root.title("FACE RECOGNITION")

#creating a text label
Label(root, text="FACE RECOGNITION USING DEEP LEARNING",font=("times new roman",20),fg="white",bg="red",height=2).grid(row=0,rowspan=2,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

#creating first button
Button(root,text="START",font=("times new roman",20),bg="#0D47A1",fg='white',command=function1).grid(row=3,columnspan=2,sticky=W+E+N+S,padx=5,pady=5)

#creating second button
Button(root,text="VERIFY",font=("times new roman",20),bg="#0D47A1",fg='white',command=function2).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="REGISTER",font=("times new roman",20),bg="#0D47A1",fg='white',command=function3).grid(row=5,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="RESET",font=("times new roman",20),bg="#0D47A1",fg='white',command=function4).grid(row=6,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

root.mainloop()

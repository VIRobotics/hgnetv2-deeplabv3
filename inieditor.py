# -*- coding : utf-8-*-
import tkinter as tk
from tkinter import filedialog
import configparser

class ConfigEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.config = configparser.ConfigParser()
        self.config.read('config.ini',encoding='utf-8')
        self.folder_path = tk.StringVar(value=self.config["base"].get("dataset_path", 'VOCdevkit'))
        self.save_folder_path = tk.StringVar(value=self.config["base"].get("save_path","save"))


        for i,l in enumerate(["数据集路径","保存路径"]):
            tk.Label( text=l).grid(row=i, column=1)

        lbl1 = tk.Entry(master=self.master, textvariable=self.folder_path, width=48)
        lbl1.grid(row=0, column=2)
        button2 = tk.Button(text="Browse", command=self.browse_button)
        button2.grid(row=0, column=3)

        lbl2 = tk.Entry(master=self.master, textvariable=self.save_folder_path, width=48)
        lbl2.grid(row=1, column=2)
        button3 = tk.Button(text="Browse", command=self.browse_button_save)
        button3.grid(row=1, column=3)

        # 创建控件
        #self.textbox = tk.Text(self.master)
        self.save_button = tk.Button( text="保存", command=self.save_config)


        # 将配置文件内容读入文本框
        for section in self.config.sections():
            #self.textbox.insert(tk.END, "[{}]\n".format(section))
            for key, value in self.config.items(section):
                pass
                #self.textbox.insert(tk.END, "{} = {}\n".format(key, value))

        # 布局控件
        # self.textbox.pack()
        self.save_button.grid(row=5, column=2)

    def save_config(self):
        # 从文本框中读取内容并写入配置文件中
        self.config["base"]["dataset_path"]=self.folder_path.get()
        self.config["base"]["save_path"] = self.save_folder_path.get()
        with open('config-new.ini', 'w',encoding='utf-8') as fp:
            self.config.write(fp)

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)
        print(filename)


    def browse_button_save(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.save_folder_path.set(filename)
        print(filename)

root = tk.Tk()
app = ConfigEditor(root)

root.mainloop()
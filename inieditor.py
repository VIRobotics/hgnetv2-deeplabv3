# -*- coding : utf-8-*-
import tkinter as tk
from tkinter import filedialog
import configparser
from tkinter import ttk

class BackboneStep(tk.Toplevel):
    def __init__(self,cfg):
        super().__init__()
        self.title('设置主干网络')
        self.cfg=cfg
        self.config = configparser.ConfigParser()
        self.config.read(cfg, encoding='utf-8')
        _bb_table={
            "unet":("resnet50","vgg","hgnetv2l"),
            "lab":("mobilenetv2","mobilenetv3s","mobilenetv3l","hgnetv2x","hgnetv2l",
                    "xception"),
            "segformer":tuple([ f"b{i}" for i in range(6)]),
            "pspnet":("resnet50","mobilenetv2")
        }
        arch=self.config["base"].get("arch", 'lab')
        for i,l in enumerate(["主干","检测头"]):
            tk.Label(self, text=l).grid(row=i, column=1)
        self.bb = ttk.Combobox(self)
        i=_bb_table[arch].index(self.config["base"].get("backbone", 'ertsd'))
        if i<0:i=0
        self.bb['value'] = _bb_table[arch]
        self.bb.current(i)
        self.bb.grid(row=0, column=2)
        if arch=="lab":
            _header=("ASPP","transformer")
        else:
            _header=("disabled",)

        self.header = ttk.Combobox(self)
        i = _header.index(self.config["base"].get("header", 'ertsd'))
        if i < 0: i = 0
        self.header['value'] = _header
        self.header.current(i)
        self.header.grid(row=1, column=2)

        self.save_button = tk.Button(self,text="完成", command=self.save_config)
        self.save_button.grid(row=2, column=2)


    def save_config(self):
        # 从文本框中读取内容并写入配置文件中
        self.config["base"]["backbone"]=self.bb.get()
        self.config["base"]["header"] = self.header.get()
        with open(self.cfg, 'w',encoding='utf-8') as fp:
            self.config.write(fp)
        self.destroy()






class ConfigEditor(tk.Frame):
    def __init__(self, master=None,ini="config.ini"):
        super().__init__(master)
        self.master = master
        self.ini=ini
        self.config = configparser.ConfigParser()
        self.config.read(ini,encoding='utf-8')
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

        self.arch=ttk.Combobox()
        arl=('unet', 'lab', 'pspnet', 'segformer')
        i=arl.index(self.config["base"].get("arch", 'lab'))
        if i<0:
            i=0
        self.arch['value'] = arl
        self.arch.current(i)
        self.arch.grid(row=2, column=2)

        # 创建控件
        #self.textbox = tk.Text(self.master)
        self.save_button = tk.Button( text="下一步", command=self.save_config)
        self.save_button.grid(row=5, column=3)

        self.close_button = tk.Button(text="结束", command=self.close)
        self.close_button.grid(row=5, column=1)
    def close(self):
        self.save_config()
        self.destroy()

    def save_config(self):
        # 从文本框中读取内容并写入配置文件中
        self.config["base"]["dataset_path"]=self.folder_path.get()
        self.config["base"]["save_path"] = self.save_folder_path.get()
        self.config["base"]["arch"] = self.arch.get()
        with open(self.ini, 'w',encoding='utf-8') as fp:
            self.config.write(fp)
        self.ask_userinfo()

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

    def ask_userinfo(self):
        inputDialog = BackboneStep(self.ini)
        self.wait_window(inputDialog)  # 这一句很重要！！！
        return
if __name__=="__main__":
    import argparse,os
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.ini")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    root = tk.Tk()
    app = ConfigEditor(root,args.config)
    root.mainloop()

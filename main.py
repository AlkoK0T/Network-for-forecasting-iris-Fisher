import numpy
import scipy.special
from tkinter import *
from tkinter import ttk
output_nodes=10
class network:
    def __init__(self, innodes, hidenodes,outnodes, learrate, rdbool):
        self.inodes=innodes
        self.hnodes=hidenodes
        self.onodes=outnodes
        self.lrate=learrate
        self.righttrainbool=0
        if (rdbool):
            self.wih = numpy.random.normal(0.0,pow(hidenodes,-0.5),(hidenodes,innodes))
            self.who = numpy.random.normal(0.0,pow(outnodes,-0.5), (outnodes, hidenodes))
        else:
            from wih import wih
            from who import who
            self.wih=numpy.array(wih)
            self.who=numpy.array(who)
        self.act_func = lambda x:scipy.special.expit(x)
        pass
    def train(self,input_list,target_list):
        inp = numpy.array(input_list,ndmin=2).T
        targ = numpy.array(target_list,ndmin=2).T
        hideninputs = numpy.dot(self.wih, inp)
        hidenout = self.act_func(hideninputs)
        finalinput = numpy.dot(self.who,hidenout)
        finalout = self.act_func(finalinput)
        outputerrors = targ-finalout
        if numpy.argmax(finalout)==numpy.argmax(targ):
            self.righttrainbool+=1
        hidenerrors = numpy.dot(self.who.T, outputerrors)
        cd=outputerrors*finalout*(1.0-finalout)
        cd1=numpy.transpose(hidenout)
        self.who += self.lrate*numpy.dot((outputerrors*finalout*(1.0-finalout)),numpy.transpose(hidenout))
        self.wih += self.lrate*numpy.dot((hidenerrors*hidenout*(1.0-hidenout)),numpy.transpose(inp))
        pass
    def query(self, input_list):
        inp = numpy.array(input_list,ndmin=2).T
        hideninputs = numpy.dot(self.wih, inp)
        hidenout = self.act_func(hideninputs)
        finalinput = numpy.dot(self.who,hidenout)
        finalout = self.act_func(finalinput)
        return finalout
    def query_image(self, input_list):
        inp = numpy.array(input_list,ndmin=2).T
        hideninputs = numpy.dot(self.wih, inp)
        hidenout = self.act_func(hideninputs)
        finalinput = numpy.dot(self.who,hidenout)
        finalout = self.act_func(finalinput)
        return finalout
    def epoch(self, file, epochs):
        root = Tk()
        root.title("Train indicator")
        root.geometry("300x150") 
        epoch_var = IntVar()
        trainset_var = IntVar()
        efficienty=IntVar()
        epochbar =  ttk.Progressbar(orient="horizontal", maximum=epochs, variable=epoch_var)
        epochbar.grid(row=1, column=1, columnspan=5)
        epochtext = ttk.Label(text="Эпоха")
        epochtext.grid(row=1, column=6)
        epochvar = ttk.Label(textvariable=epoch_var)
        epochvar.grid(row=1, column=7)
        trainsetbar =  ttk.Progressbar(orient="horizontal", maximum=len(file), variable=trainset_var)
        trainsetbar.grid(row=2, column=1, columnspan=5)
        trainsettext = ttk.Label(text="Тренировочный")
        trainsettext.grid(row=2, column=6)
        trainsetvar = ttk.Label(textvariable=trainset_var)
        trainsetvar.grid(row=2, column=7) 
        effitext = ttk.Label(text="Эффективность")
        effitext.grid(row=3, column=1)
        effivar = ttk.Label(textvariable=efficienty)
        effivar.grid(row=3, column=2) 
        for i in range(epochs):
            for rec in file:
                all_val=rec.split(',')
                inputs = (numpy.asarray(all_val[1:],dtype=float) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(output_nodes) +0.01
                targets[int(all_val[0])] =0.99
                n.train(inputs,targets)
                trainsetbar.step(1)
                efficienty=self.righttrainbool/trainset_var.get()
                root.update()
                pass
            wih=open("wih.py",'w')
            wih.write("wih= " + str(self.wih.tolist()))
            wih.close()
            who=open("who.py",'w')
            who.write("who="+str(self.who.tolist()))
            who.close()
            epochbar.step(1)
            root.update()
        root.mainloop()
n = network(784,200,10,0.1,True)
tdf = open("iris.csv", 'r')
tdl = tdf.readlines()
tdl = tdl[1:]
tdf.close()
n.epoch(tdl,5)
tedf = open("iris.csv", 'r')
tedl = tedf.readlines()
tedl = tedl[1:]
tedf.close()
pr=0
scorecard=[]
for i in range(5):
    for rec in tedl:
        all_val=rec.split(',')
        inputs = (numpy.asarray(all_val[1:],dtype=float) / 255.0 * 0.99) + 0.01
        out=n.query(inputs)
        if numpy.argmax(out)==int(all_val[0]):
            scorecard.append(1)
        else:
            scorecard.append(0)
        pass
scorecard_array = numpy.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)
fd=0
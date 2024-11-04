import numpy
class network:
    def __init__(self, innodes, hidenodes, outnodes, lrate, epoch):
        self.inn=innodes
        self.hin=hidenodes
        self.oun=outnodes
        self.lr=lrate
        self.ep=epoch
        
        # Определяем веса
        self.wih=[]

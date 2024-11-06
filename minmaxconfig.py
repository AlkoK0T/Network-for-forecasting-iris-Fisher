class minmax:
    def __init__(self):
        self.tdf = open("iris.csv", 'r')
        self.tdl =self.tdf.readlines()[1:]
        self.tdlsplit=[line.split(',') for line in self.tdl]
        self.maxlist = []
        self.minlist = []
        for i in range(1,len(self.tdlsplit[0])-1):
            self.maxlist.append(max([float(j[i]) for j in self.tdlsplit]))
            self.minlist.append(min([float(j[i]) for j in self.tdlsplit]))
    def region(self):
        self.SepalLengthCm = [float(i.split(',')[1]) for i in self.tdl]
        self.SepalWidthCm = [float(i.split(',')[2]) for i in self.tdl]
        self.PetalLengthCm = [float(i.split(',')[3]) for i in self.tdl]
        self.PetalWidthCm = [float(i.split(',')[4]) for i in self.tdl]
        self.SepalLengthCmmin=min(self.SepalLengthCm)
        self.SepalLengthCmmax=max(self.SepalLengthCm)
        self.SepalWidthCmmin=min(self.SepalWidthCm)
        self.SepalWidthCmmax=max(self.SepalWidthCm)
        self.PetalLengthCmmin=min(self.PetalLengthCm)
        self.PetalLengthCmmax=max(self.PetalLengthCm)
        self.PetalWidthCmmin=min(self.PetalWidthCm)
        self.PetalWidthCmmax=max(self.PetalWidthCm)
        pass
    def coefficient(self,num,cur):
        return (cur-self.minlist[num])*2/(self.maxlist[num] - self.minlist[num])+(-1)

c=minmax()
d=c.region()
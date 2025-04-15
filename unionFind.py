class UnionFind:
    def __init__(self,numPoints):
        self.li = [0] * numPoints
        for i in range(numPoints):
            self.li[i] = i
        self.n_components = numPoints

    def find(self,j):
        idx = j
        while self.li[idx] != idx:
            nextIdx = self.li[idx]
            self.li[idx] = self.li[nextIdx]
            idx = nextIdx
        return idx

    def union(self,i,j):
        f1 = self.find(i)
        f2 = self.find(j)
        if f1 != f2:
            self.li[f1] = f2
            self.n_components -= 1

if __name__ == "__main__":
    uf = UnionFind(5)
    uf.union(0,1)
    uf.union(0,2)
    uf.union(1,2)
    uf.union(3,4)

    for i in range(5):
        print(uf.find(i))

    print(uf.n_components)
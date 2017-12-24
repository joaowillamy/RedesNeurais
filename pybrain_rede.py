from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(2, 1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

trainer = BackpropTrainer(net, ds)
trainer.train()
# trainer.trainUntilConvergence()

net.activate([2, 1])
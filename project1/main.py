from project1.ibm import IBM

ibm1 = IBM('IBM1',
           './data/training/hansards.36.2.e',
           './data/training/hansards.36.2.f',
           './data/validation/dev.e',
           './data/validation/dev.f',
           './data/validation/dev.wa.nonullalign')


ibm2 = IBM('IBM2',
           './data/training/hansards.36.2.e',
           './data/training/hansards.36.2.f',
           './data/validation/dev.e',
           './data/validation/dev.f',
           './data/validation/dev.wa.nonullalign')

ibm1.train(3)
# ibm2.train(3)


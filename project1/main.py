from project1.ibm1 import IBM1

ibm1 = IBM1('./data/training/hansards.36.2.e',
            './data/training/hansards.36.2.f',
            './data/validation/dev.e',
            './data/validation/dev.f',
            './data/validation/dev.wa.nonullalign')

ibm1.train(1)


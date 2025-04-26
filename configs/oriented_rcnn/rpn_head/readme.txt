## Introduction
xxx_test1/2 ： 测试生成不同长宽比的中点偏移表示框
xxx——test_loss/loss_e:测试长宽比加权的loss，分别是weight=(2d+2)/(2+d)和weight=2 - e^(-1*d)
xxx_pconv/ARconv:测试在head的rpn卷积后加一个卷积，
arc_r50，pc_r50：分别是用ar卷积和pc卷积替换resnet的一部分，相当于换了主干
arc = ad , means : adaptive rotate conv
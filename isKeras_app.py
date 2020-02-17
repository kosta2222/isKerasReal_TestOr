import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
eps=10
x=[[1,1],[1,0],[0,1],[0,0] ]
y=[[1],[1],[1],[0]]

vec_x=[0]*2*4
vec_y=[0]*1*4
def get_x_data():
    global vec_x
    for r in range(4):
        for e in range(2):
            val=x[r][e]
            vec_x[r*2+e]=val
def get_y_data():
    global vec_y
    for r in range(4):
        for e in range(1):
            val=y[r][e]
            vec_y[r*1+e]=val

class NN_pars:
    inputNeurons=None
    outputsNeurons=None
NN=NN_pars()
NN.inputNeurons=2
NN.outputNeurons=1
max_in_nn=2
max_rows_orOut=1
max_validSet_rows=4
def cross_validation(model,X_test:list,Y_test:list,rows,cols_X_test,cols_Y_test)->float:
    tmp_vec_x_test=[0]*max_in_nn
    tmp_vec_y_test=[0]*max_rows_orOut
    scores=[0]*max_validSet_rows
    index_row=0
    res=0
    out_NN=np.array((1,NN.outputNeurons))
    res_acc=0
    for row in range(rows):
        for elem1 in range(NN.inputNeurons):
            tmp_vec_x_test[elem1]=X_test[row*cols_X_test+elem1]
        for elem2 in range(NN.outputNeurons):
            tmp_vec_y_test[elem2]=Y_test[row*cols_Y_test+elem2]
        # predicr в out_NN
        out_NN=model.predict(np.array([tmp_vec_x_test]))
        res=check_2oneHotVecs(out_NN.tolist()[0],tmp_vec_y_test,NN.outputNeurons)
        scores[index_row]=res
        res=0
        index_row+=1
    res_acc=calc_accur(scores,rows)
    print("Ac:%f%s"%(res_acc,"%"))
    return res_acc
def check_2oneHotVecs(out_NN:list,vec_y_test,vec_size)->int:
    tmp_elemOf_outNN_asHot=0
    for col in range(vec_size):
        tmp_elemOf_outNN_asHot=out_NN[col]
        if (tmp_elemOf_outNN_asHot>0.5):
              tmp_elemOf_outNN_asHot=1
        else:
            tmp_elemOf_outNN_asHot=0
        if(tmp_elemOf_outNN_asHot==int(vec_y_test[col])):
          continue
        else:
          return 0
    return 1    
def calc_accur(scores:list,rows)->float:
    accuracy=0
    sum=0
    for col in range(rows):
        sum+=scores[col]
    accuracy=sum/rows*100
    return accuracy
def plot_history(history):
    fig,ax=plt.subplots()
    x=range(eps)
    plt.plot(x,history.history['loss'])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mse")
    plt.show()
def main():
   print("Keras learns") 
   model=Sequential()
   model.add(Dense(3,input_dim=2,activation='sigmoid',use_bias=True))
   model.add(Dense(1,activation='sigmoid',use_bias=True))
   model.compile(optimizer=SGD(lr=0.7),loss='mse',metrics=['accuracy'])
   history=model.fit(np.array(x),np.array(y),validation_data=(np.array(x),np.array(y)),epochs=eps)
   plot_history(history)
   print("End of learning")
   print("My cross validation")
   get_x_data()
   get_y_data()
   cross_validation(model,vec_x,vec_y,4,2,1)

main()



```python
# google's own was a little too complex for beginners, so this is a watered down version to promote core understanding#
# ===============implementation of a seq2seq machine translator from german to english======================================#
# ================input (source_data) is in german and output(target is in english)=========================================#
# ================ we are operating on time major basis menaning we recieve out answers with time as its first dimension ====================================================================#
# ================ implemented by shuaibu Alexander =====================================================================#
```


```python
# imports
import numpy as np
import tensorflow as tf
```


```python
# defination of global variables
number_of_sentences_to_read=50000
max_input_length=41
enc_time_step=41
dec_time_step=61
batch_size=18
num_units=128
max_output_length=61
vocabulary_size=50000
decoder_type="basic"
```


```python
# step 1- read in in of words vocabularies from the raw-dataset

source_dictionary=dict()
target_dictionary=dict()
# read in the vocabulary of the source
with open(r"seq2seq\vocab.50k.de",encoding="utf-8") as f:
    for source_row in f:
        source_dictionary[source_row[:-1]]=len(source_dictionary)# indexing up to -1 because it ends with a new line we dont want
reverse_source_dictionary=dict(zip(source_dictionary.values(),source_dictionary.keys()))

print("\nread in source vocabulary")
print(list(source_dictionary.items())[:10])
print(list(reverse_source_dictionary.items())[:10])
print("vocabulary length:",len(source_dictionary))
        
# read in the vocabulary of the target
with open(r"seq2seq\vocab.50k.en",encoding="utf-8") as f:
    for target_row in f:
        target_dictionary[target_row[:-1]]=len(target_dictionary)# indexing up to -1 because it ends with a new line we dont want
reverse_target_dictionary=dict(zip(target_dictionary.values(),target_dictionary.keys()))

print("\nread in target vocabulary")
print(list(target_dictionary.items())[:10])
print(list(reverse_target_dictionary.items())[:10])
print("vocabulary length:",len(target_dictionary))
```

    
    read in source vocabulary
    [('<unk>', 0), ('<s>', 1), ('</s>', 2), (',', 3), ('.', 4), ('die', 5), ('der', 6), ('und', 7), ('in', 8), ('zu', 9)]
    [(0, '<unk>'), (1, '<s>'), (2, '</s>'), (3, ','), (4, '.'), (5, 'die'), (6, 'der'), (7, 'und'), (8, 'in'), (9, 'zu')]
    vocabulary length: 50000
    
    read in target vocabulary
    [('<unk>', 0), ('<s>', 1), ('</s>', 2), ('the', 3), (',', 4), ('.', 5), ('of', 6), ('and', 7), ('to', 8), ('in', 9)]
    [(0, '<unk>'), (1, '<s>'), (2, '</s>'), (3, 'the'), (4, ','), (5, '.'), (6, 'of'), (7, 'and'), (8, 'to'), (9, 'in')]
    vocabulary length: 50000
    


```python
# step 2- read in the dataset for both the source and target sequence
# input->german output(target)->english
input_dataset=[]
output_dataset=[]

with open(r"seq2seq\train.de",encoding="utf-8") as f:
    for index_of_row_read,input_row in enumerate(f):
        if index_of_row_read<50: #because there are some errors in translation for the first few lines
            continue
        if index_of_row_read>number_of_sentences_to_read:
            break
        input_dataset.append(input_row)

        
with open(r"seq2seq\train.en",encoding="utf-8") as f:
    for index_of_row_read,target_row in enumerate(f):
        if index_of_row_read<50:
            continue
        if index_of_row_read>number_of_sentences_to_read:
            break
        output_dataset.append(target_row)

print("example translations:")
for i in range(0,number_of_sentences_to_read,10000):
    print("",i,":",input_dataset[i],"\n")
    print("",i,":",output_dataset[i],"\n")
```

    example translations:
     0 : Heute verstehen sich QuarkXPress ® 8 , Photoshop ® und Illustrator ® besser als jemals zuvor . Dank HTML und CSS ­ können Anwender von QuarkXPress inzwischen alle Medien bedienen , und das unabhängig von Anwendungen der Adobe ® Creative Suite ® wie Adobe Flash ® ( SWF ) und Adobe Dreamweaver ® .
     
    
     0 : Today , QuarkXPress ® 8 has tighter integration with Photoshop ® and Illustrator ® than ever before , and through standards like HTML and CSS , QuarkXPress users can publish across media both independently and alongside Adobe ® Creative Suite ® applications like Adobe Flash ® ( SWF ) and Adobe Dreamweaver ® .
     
    
     10000 : Es existieren Busverbindungen in nahezu jeden Ort der Provence ( eventuell mit Umsteigen in Aix ##AT##-##AT## en ##AT##-##AT## Provence ) , allerdings sollte beachtet werden , dass die letzten Busse abends ca. um 19 Uhr fahren .
     
    
     10000 : As always in France those highways are expensive but practical , comfortable and fast .
     
    
     20000 : Es war staubig , das Bad schmutzig . Sogar die Beleuchtung an der Wand im Flur ( Seitengebäude ) war richtig verstaubt .
     
    
     20000 : It was rather old fashioned in the decoration .
     
    
     30000 : Auch ist , so denkt Dr. Gutherz , bereits die erste Seite sehr viel versprechend , da sie eine Definition des klinischen Psychotrauma ##AT##-##AT## Begriffes enthält , der er gänzlich zustimmen kann .
     
    
     30000 : At the rhetorical climax of this summary , Dr Goodheart comes across some sentences expressed with great pathos .
     
    
     40000 : Bei einer digitalen Bildkette wird das Intensitätssignal für jedes Pixel ohne analoge Zwischenschritte direkt in der Detektoreinheit digitalisiert , d.h. in Zahlen umgewandelt .
     
    
     40000 : A digital image chain is an image chain that is equipped with a digital detector instead of an analogue one .
     
    
    


```python
# step-4 tokenize the word
def tokenize_sentence(sentence,is_source):
    sentence=sentence.replace("."," .")
    sentence=sentence.replace(","," ,")
    sentence=sentence.replace("\n"," ")
    
    sentence_tokens=sentence.split()
    
    for index,token in enumerate(sentence_tokens):
        if is_source:
            if token not in source_dictionary:
                sentence_tokens[index]="<unk>"
        else:
            if token not in target_dictionary:
                sentence_tokens[index]="<unk>"
    return sentence_tokens

# computing some statistics
source_lengths=[]
target_lengths=[]

for source_word,target_word in zip(input_dataset,output_dataset):
    source_lengths.append(len(tokenize_sentence(source_word,True)))
    target_lengths.append(len(tokenize_sentence(target_word,False)))
    
print("\nSource")
print("average length",np.mean(source_lengths))
print("standard deviation",np.std(source_lengths))

print("\nTarget")
print("average length",np.mean(target_lengths))
print("standard deviation",np.std(target_lengths))

```

    
    Source
    average length 23.1045624712
    standard deviation 12.6389970633
    
    Target
    average length 25.3302236191
    standard deviation 13.8682905263
    


```python
# step-4 preprocess the text and add extra pad tokens to make them of equal lengths
print("preprocessing the data")
train_inputs=[]
train_outputs=[]

train_inputs_lengths=[]
train_outputs_lengths=[]


for train_input,train_output in zip(input_dataset,output_dataset):
    num_train_input,num_train_output=[],[]
    tokenized_input=tokenize_sentence(train_input,True)
    tokenized_output=tokenize_sentence(train_output,False)
    
    #for input data     
    for token in tokenized_input:
        num_train_input.append(source_dictionary[token])
    num_train_input.insert(0,source_dictionary["<s>"])
    
    train_inputs_lengths.append(min(len(num_train_input)+1,max_input_length))
    if len(num_train_input)+1<max_input_length:
        num_train_input.extend([source_dictionary["</s>"]]*(max_input_length-(len(tokenized_input)+1)))
        
    elif len(num_train_input)+1<max_input_length:
        num_train_input=num_train_input[:max_input_length]
       
    train_inputs.append(num_train_input)
#     for output data
    for token in tokenized_output:
        num_train_output.append(target_dictionary[token])
    num_train_output.insert(0,target_dictionary["</s>"])
    
    train_outputs_lengths.append(min(len(num_train_output)+1,max_output_length))
    if len(num_train_output)+1<max_output_length:
        num_train_output.extend([target_dictionary["</s>"]]*(max_output_length-(len(tokenized_output)+1)))
        
    elif len(num_train_output)+1>max_output_length:
        num_train_output=num_train_output[:max_output_length]
        
    train_outputs.append(num_train_output)
print("sample data")
print(" ".join([reverse_source_dictionary[w] for w in train_inputs[500]])+"\tof length",train_inputs_lengths[500])
print(" ".join([reverse_target_dictionary[w] for w in train_outputs[500]])+ "\tof length",train_outputs_lengths[500])
```

    preprocessing the data
    sample data
    <s> Je mehr Zeit wir mit Gilad und dem Rest des Teams in Israel verbracht haben ( um nicht den lauten Hahn zu erwähnen der <unk> bei denen über den Campus <unk> ) desto überzeugter waren wir – zusammen können wir mehr bewegen .	of length 41
    </s> The more time we spent with Gilad as well as the rest of the team in Israel ( not to mention the very loud <unk> that runs around in their campus ) , the more convinced we all became - we ’ ll be better off together . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>	of length 50
    


```python
# step-5 create a data batch generator which uses time major i.e [time_step*batch_size]

class DataGeneratorMT(object):
    def __init__(self,batch_size,is_source):
        self._batch_size=batch_size
        self._cursors=[0]*batch_size
        self._is_source=is_source
        if is_source:
            self._max_length=41
        else:
            self._max_length=61
        
    def next_batch(self,data_indexes):
        batch_output=[0]*self._batch_size
        batch_label=[0]*self._batch_size
        
        for i,data_index in enumerate(data_indexes):
            if self._is_source:
                sent_text=train_inputs[data_index]
            else:
                sent_text=train_outputs[data_index]
            batch_output[i]=sent_text[self._cursors[i]]
            batch_label[i]=sent_text[self._cursors[i]+1]
            
            self._cursors[i]=(self._cursors[i]+1)%(self._max_length-2)
        return batch_output,batch_label
    
    def reset_cursor(self):
        self._cursors=[0]*self._batch_size
        
    def unroll_batches(self,unroll_length,data_indexes):
        self.reset_cursor()
        batch_outputs,batch_labels,batch_lengths=[],[],[]
        
        for ui in range(unroll_length):
            batch_output,batch_label=self.next_batch(data_indexes)
            batch_outputs.append(batch_output)
            batch_labels.append(batch_label)
        
        if self._is_source:
            batch_lengths=np.array(train_inputs_lengths)[data_indexes]
        else:
            batch_lengths=np.array(train_outputs_lengths)[data_indexes]
        
        return batch_outputs,batch_labels,data_indexes,batch_lengths
    
print("sample generated data\n")
dt=DataGeneratorMT(1,False)
data,_,_,_=dt.unroll_batches(60,[0])
print(" ".join([reverse_target_dictionary[datum] for datum in np.array(data).reshape(-1)]))

print()
dt=DataGeneratorMT(1,True)
data,_,_,_=dt.unroll_batches(60,[0])
print(" ".join([reverse_source_dictionary[datum] for datum in np.array(data).reshape(-1)]))
```

    sample generated data
    
    </s> Today , QuarkXPress ® 8 has tighter integration with Photoshop ® and Illustrator ® than ever before , and through standards like HTML and CSS , QuarkXPress users can publish across media both independently and alongside Adobe ® Creative Suite ® applications like Adobe Flash ® ( SWF ) and Adobe Dreamweaver ® . </s> </s> </s> </s> </s>
    
    <s> Heute verstehen sich QuarkXPress ® 8 , Photoshop ® und Illustrator ® besser als jemals zuvor . Dank HTML und CSS ­ können Anwender von QuarkXPress inzwischen alle Medien bedienen , und das unabhängig von Anwendungen der Adobe <s> Heute verstehen sich QuarkXPress ® 8 , Photoshop ® und Illustrator ® besser als jemals zuvor . Dank HTML und
    


```python
# constuction of placeholders of the graph
tf.reset_default_graph()
with tf.name_scope("input_placeholders"):
    enc_train_inputs=tf.placeholder(shape=[enc_time_step,batch_size],dtype=tf.int32)
    enc_sequence_lengths=tf.placeholder(shape=[batch_size],dtype=tf.int32)
    dec_train_inputs=tf.placeholder(shape=[dec_time_step,batch_size],dtype=tf.int32)
    dec_train_labels=tf.placeholder(shape=[dec_time_step,batch_size],dtype=tf.int32)
    dec_labels_mask=tf.placeholder(shape=[dec_time_step,batch_size],dtype=tf.float32)

with tf.name_scope("embeddings"):
    enc_embeddings=tf.convert_to_tensor(np.load(r"./seq2seq/de-embeddings.npy"))
    dec_embeddings=tf.convert_to_tensor(np.load(r"./seq2seq/en-embeddings.npy"))

with tf.name_scope("input_variables"):
    enc_emb_inputs=tf.nn.embedding_lookup(enc_embeddings,enc_train_inputs)
    dec_emb_inputs=tf.nn.embedding_lookup(dec_embeddings,dec_train_inputs)
```


```python
# defination of encoder
basic_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)

initial_state=basic_cell.zero_state(batch_size,dtype=tf.float32)

encoder_outputs,encoder_states=tf.nn.dynamic_rnn(
    basic_cell,enc_emb_inputs,initial_state=initial_state
    ,dtype=tf.float32,time_major=True,sequence_length=enc_sequence_lengths
)
```


```python
# defination of decoder
basic_decoder_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
training_helper=tf.contrib.seq2seq.TrainingHelper(dec_emb_inputs,sequence_length=[max_output_length]*batch_size,time_major=True)
output_layer=tf.layers.Dense(units=vocabulary_size)

if decoder_type=="basic":
    decoder=tf.contrib.seq2seq.BasicDecoder(basic_decoder_cell,training_helper,initial_state=encoder_states,output_layer=output_layer)
    
if decoder_type=="attention":
    decoder=tf.contrib.seq2seq.BahdanauAttention(basic_decoder_cell,training_helper,initial_state=encoder_states,output_layer=output_layer)

outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=True,swap_memory=True)
prediction=outputs.sample_id
```


```python
# loss 
xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=dec_train_labels,logits=outputs.rnn_output
)
loss=(tf.reduce_sum(xentropy*dec_labels_mask))/(batch_size*61)
prediction=outputs.sample_id
```


```python
# optimization

# creating a learning rate decay
global_step=tf.Variable(0,trainable=False)
inc_global_step=tf.assign(global_step,global_step+1)
learning_rate=tf.train.exponential_decay(0.001,global_step,decay_steps=10,decay_rate=0.9,staircase=True)

# using two different optimization algorithms
# using gradient clipping to prevent gradient explosion
adam_optimizer=tf.train.AdamOptimizer(learning_rate)
adam_grad,variable_name=zip(*adam_optimizer.compute_gradients(loss))
clipped_grad,_=tf.clip_by_global_norm(adam_grad,25.0)
adam_optimize=adam_optimizer.apply_gradients(zip(clipped_grad,variable_name))

sgd_optimizer=tf.train.GradientDescentOptimizer(learning_rate)
sgd_grad,variable_name1=zip(*sgd_optimizer.compute_gradients(loss))
clipped_grad1,_=tf.clip_by_global_norm(sgd_grad,25.0)
sgd_optimize=sgd_optimizer.apply_gradients(zip(clipped_grad1,variable_name1))

sess=tf.InteractiveSession()
```

    C:\Users\Administrator\Anaconda3\lib\site-packages\tensorflow\python\client\session.py:1711: UserWarning: An interactive session is already active. This can cause out-of-memory errors in some cases. You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).
      warnings.warn('An interactive session is already active. This can '
    


```python
# training and infering
num_steps=10001
avg_loss=0

enc_data_generator=DataGeneratorMT(batch_size,True)
dec_data_generator=DataGeneratorMT(batch_size,False)
print("started_training")
sess.run(tf.global_variables_initializer())
for step in range(10):
    
    #     defining the batch data.
    batch_indexes=np.random.choice(train_inputs[1],batch_size)
    enc_data,_,_,enc_lengths=enc_data_generator.unroll_batches(41,batch_indexes)
    dec_data,dec_labels,_,dec_lengths=dec_data_generator.unroll_batches(61,batch_indexes)
    dec_masks=np.zeros(shape=np.array(dec_data).shape)
    #     getting the masks
    for i,data in enumerate(dec_data):
        dec_masks[i]=np.array([i for _ in range(batch_size)])<dec_lengths
    
    #     feeding in the placeholders.
    feed_dict={
        enc_train_inputs:enc_data,
        dec_train_inputs:dec_data,
        dec_train_labels:dec_labels,
        enc_sequence_lengths:enc_lengths,
        dec_labels_mask:dec_masks
    }
    
    
    if step<1000:
        _,loss_value,predicted_value=sess.run([adam_optimize,loss,prediction],feed_dict=feed_dict)
    else:
        _,loss_value,predicted_value=sess.run([sgd_optimize,loss,prediction],feed_dict=feed_dict)
    
    print("\t\t\tstep:{},loss:{}".format(step,loss_value))
    predicted_value=predicted_value.flatten()
    avg_loss+=loss_value
    
    if not (step+1)%100:
        print("\t\t step:",step,"\t\t")
        
        print_str="Actual"
        
        for word in np.concatenate(dec_labels,0)[::batch_size]:
            print_str+=reverse_target_dictionary[word]+" "
            if reverse_target_dictionary[word]=="</s>":
                break
        print(print_str)
        
        print("\n")
        for word in predicted_value[::batch_size]:
            print_str+=reverse_target_dictionary[word]+" "
            if reverse_target_dictionary[word]=="</s>":
                break
        print(print_str)
        
        print("======================= average loss ====================") 
        print(avg_loss/100)
        
        loss_over_time.append(avg_loss/100)
        sess.run(inc_global_step)
        avg_loss=0.0
    
        
```

    started_training
    			step:0,loss:4.168298244476318
    			step:1,loss:4.265270233154297
    			step:2,loss:4.263433456420898
    			step:3,loss:4.27157735824585
    			step:4,loss:4.280276298522949
    			step:5,loss:4.26577091217041
    			step:6,loss:4.260280132293701
    			step:7,loss:4.570485591888428
    			step:8,loss:4.359394550323486
    			step:9,loss:4.138807773590088
    


```python

```


```python

```

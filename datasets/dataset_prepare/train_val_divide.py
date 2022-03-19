
import os
import random
Root = '/media/E/ht/dataset/HT21'
dataset = 'HT21'

dst_imgs_path = os.path.join(Root,'images')

def divide_dataset(val_ration =0.1):
    test_set = []
    val_set= []
    train_set=[]
    train_path = os.path.join(Root+'/train')
    scenes= os.listdir(train_path)

    for i_scene in scenes:
        sub_files = os.listdir(os.path.join(train_path, i_scene+'/img1'))
        for i in sub_files:
            train_set.append(os.path.join('train/'+i_scene+'/img1',i))
    
    
    train_path = os.path.join(Root+'/test')
    scenes= os.listdir(train_path)
 
    for i_scene in scenes:
        sub_files = os.listdir(os.path.join(train_path, i_scene+'/img1'))
        for i in sub_files:
            test_set.append(os.path.join('test/'+i_scene+'/img1',i))



    print("test_set_num:", len(train_set), 'train_val_num:',len(test_set))

    # val_set = random.sample(train_set, round(val_ration * len(train_val)))
    print("val_set_num:", len(val_set))
    train_set = set(train_set)
    val_set   = set(val_set)
    train_set = train_set - val_set
    print("train_set_num:", len(train_set))

    train_set = sorted(train_set)
    val_set = sorted(val_set)
    test_set = sorted(test_set)

    with open(os.path.join(Root,'train.txt'), "w") as f:
        for train_name in train_set:
            f.write(train_name+'\n')
    f.close()

    with open(os.path.join(Root,'val.txt'), "w") as f:
        for valid_name in val_set:
            f.write(valid_name+'\n')

    f.close()

    with open(os.path.join(Root,'test.txt'), "w") as f:
        for test_name in test_set:
            f.write(test_name+'\n')

    f.close()


def divide_dataset(val_ration=0.1):
    test_set = []
    val_set = []
    train_set = []
    train_path = os.path.join(Root + '/train')
    scenes = os.listdir(train_path)

    for i_scene in scenes:
            train_set.append(os.path.join('train/' + i_scene))

    train_path = os.path.join(Root + '/test')
    scenes = os.listdir(train_path)

    for i_scene in scenes:
        test_set.append(os.path.join('test/' + i_scene ))

    print("test_set_num:", len(train_set), 'train_val_num:', len(test_set))

    # val_set = random.sample(train_set, round(val_ration * len(train_val)))
    print("val_set_num:", len(val_set))
    train_set = set(train_set)
    val_set = set(val_set)
    train_set = train_set - val_set
    print("train_set_num:", len(train_set))

    train_set = sorted(train_set)
    val_set = sorted(val_set)
    test_set = sorted(test_set)

    with open(os.path.join(Root, 'train.txt'), "w") as f:
        for train_name in train_set:
            f.write(train_name + '\n')
    f.close()

    with open(os.path.join(Root, 'val.txt'), "w") as f:
        for valid_name in val_set:
            f.write(valid_name + '\n')

    f.close()

    with open(os.path.join(Root, 'test.txt'), "w") as f:
        for test_name in test_set:
            f.write(test_name + '\n')

    f.close()
if __name__ == '__main__':
    divide_dataset()
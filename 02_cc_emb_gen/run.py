import os
from engine import Engine

if __name__ == '__main__':
    ### the definitions of the following parameters can be found in engine.py

    lr = 2e-5
    batch_size = 32
    gpu = "0,1"
    sources1 = [
        'cnn',
        'huff',
        'nyt'
    ]
    sources2 = [
        'fox',
        'breit',
        'nyp'
    ]
    unfinetuned = 1
    init_train = 0
    plotting = 1
    shuffle = 0
    polarizations = [
        'emb',
        'lo',
        # 'gt',
        # 'emb_doc'
    ]
    epochs = 30

    for source1 in sources1:
        for source2 in sources2:

            for polarization in polarizations:
                # filename = f'result_{source1}_{source2}_unfinetuned={unfinetuned}.txt'

                command = f'python -u engine.py --lr={lr} --batch_size={batch_size} ' \
                          f'--source1 {source1} --source2 {source2} --gpu={gpu} --unfinetuned={unfinetuned} ' \
                          f'--init_train={init_train} --plotting={plotting} --polarization={polarization} ' \
                          f'--shuffle={shuffle} ' \
                          f'--epochs={epochs}' \
                          # f'>{filename}'

                print(command)
                os.system(command)

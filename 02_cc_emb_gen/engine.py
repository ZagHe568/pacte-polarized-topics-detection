from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from leaveout_polarization import get_leaveout_score


class NewsDataset(Dataset):
    def __init__(self, texts, labels, topic_masks):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding=True)
        self.encodings = encodings
        self.labels = labels
        self.topic_masks = topic_masks

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['topic_masks'] = torch.tensor(self.topic_masks[idx])
        return item

    def __len__(self):
        return len(self.labels)


class FC(nn.Module):
    def __init__(self, n_in, n_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


class Engine:
    def __init__(self, args):
        # gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)

        # dataset
        print('Loading data....')
        texts_processed = pd.Series(pickle.load(open(os.path.join(args.data_path, 'texts_processed_bert.pkl'), 'rb')))
        texts = texts_processed.apply(lambda x: ' '.join(x))
        df_news = pd.read_csv(os.path.join(args.data_path, 'df_news.csv'))
        # only making the model differentiate between left and right
        labels = df_news['source'].map({'cnn': 0, 'fox': 1, 'huff': 0, 'breit': 1, 'nyt': 0, 'nyp': 1})
        if args.shuffle: # shuffle the labels to serve as the baseline, where the languge model cannot learn partisanship
            labels = labels.sample(frac=1)
        del df_news
        topic_masks = pd.Series(pickle.load(open(os.path.join(args.data_path, 'topic_masks.pkl'), 'rb')))
        val_idexes = pickle.load(open(os.path.join(args.data_path, 'idxes_val.pkl'), 'rb'))
        train_idexes = set(list(range(len(texts_processed)))) - val_idexes
        # train_idexes = range(len(texts_processed))
        # val_idexes = range(len(texts_processed))

        train_idexes = np.array(list(train_idexes))
        val_idexes = np.array(list(val_idexes))
        train_mask = np.isin(np.arange(len(texts_processed)), train_idexes)
        val_mask = np.isin(np.arange(len(texts_processed)), val_idexes)
        print('Done.')

        texts_train = texts[train_mask].tolist()
        texts_val = texts[val_mask].tolist()
        texts = texts.tolist()
        labels_train = labels[train_mask].tolist()
        labels_val = labels[val_mask].tolist()
        labels = labels.tolist()
        topic_masks_train = topic_masks[train_mask].tolist()
        topic_masks_val = topic_masks[val_mask].tolist()
        topic_masks = topic_masks.tolist()
        print('Done\n')

        if args.init_train:
            print('Preparing dataset....')
            train_dataset = NewsDataset(texts_train, labels_train, topic_masks_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataset = NewsDataset(texts_val, labels_val, topic_masks_val)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            dataset = NewsDataset(texts, labels, topic_masks)
            loader = DataLoader(dataset, batch_size=int(1.5*args.batch_size))

            print('Done\n')

            # model
            print('Initializing model....')
            from transformers import AutoModelForSequenceClassification, AdamW
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

            print('Done\n')
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        os.makedirs('ckp', exist_ok=True)

        if not args.shuffle:
            model_path = f'ckp/model.pt'
        else:
            model_path = f'ckp/model_shuffle.pt'

        self.device = device
        if args.init_train:
            self.model = model
            self.optimizer = optimizer
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.loader = loader
        self.texts = texts
        self.labels = labels
        self.model_path = model_path
        self.args = args

    def train(self):

        if (not os.path.exists(self.model_path)) and (not self.args.unfinetuned):
            best_epoch_loss = float('inf')
            best_epoch_f1 = 0
            best_epoch = 0
            import copy
            best_state_dict = copy.deepcopy(self.model.state_dict())
            for epoch in range(self.args.epochs):
                print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
                loss = self.train_epoch()
                acc, f1 = self.eval()

                if f1 > best_epoch_f1:
                    best_epoch = epoch
                    best_epoch_loss = loss
                    best_epoch_f1 = f1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                print(
                    f'Epoch {epoch + 1}, Loss: {loss:.3f}, Acc: {acc:.3f}, F1: {f1:.3f}, '
                    f'Best Epoch:{best_epoch + 1}, '
                    f'Best Epoch F1: {best_epoch_f1:.3f}\n')

                if epoch - best_epoch >= 5:
                    break

            print('Saving the best checkpoint....')
            torch.save(best_state_dict, self.model_path)
            print(
                f'Best Epoch: {best_epoch + 1}, Best Epoch F1: {best_epoch_f1:.3f}, Best Epoch Loss: {best_epoch_loss:.3f}')
        self.calc_embeddings(True)
        self.calc_embeddings()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            loss = outputs[0].mean()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if i % (len(self.train_loader) // 20) == 0:
                print(f'Batch: {i + 1}/{len(self.train_loader)}\tloss:{loss.item():.3f}')

        return epoch_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()
        y_pred = []
        y_true = []
        print('Evaluating f1....')
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                y_pred.append(outputs[1].detach().to('cpu').argmax(dim=1).numpy())
                y_true.append(labels.detach().to('cpu').numpy())
                if i % (len(self.val_loader) // 10) == 0:
                    print(f"{i}/{len(self.val_loader)}")
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, f1

    def calc_embeddings(self, topic_emb=False):
        '''
        Calculate the embeddings of all documents
        :param topic_emb: boolean. If False, then the document embedding is the original BERT embedding ([CLS] embedding)
            If True, the document embedding is the document-contextualized topic embedding, with more focus on the topic keywords.
        :return: the embeddings of all documents
        '''

        os.makedirs('embeddings', exist_ok=True)
        if not topic_emb:
            embedding_path = f'embeddings/embeddings_unfinetuned={self.args.unfinetuned}.pkl'
        else:
            embedding_path = f'embeddings/topic_embeddings_unfinetuned={self.args.unfinetuned}.pkl'
        if self.args.shuffle:
            embedding_path = embedding_path[:-4] + '_shuffle.pkl'
        if not os.path.exists(embedding_path):
            if not self.args.unfinetuned:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            embeddings = []
            self.model.eval()
            print('Calculating embedding....')
            with torch.no_grad():
                for i, batch in enumerate(self.loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    if not topic_emb:
                        embeddings_ = outputs[1][-1][:, 0].detach().to('cpu').numpy()
                    else:
                        topic_masks = batch['topic_masks'].to(self.device).reshape(input_ids.shape[0],
                                                                                   input_ids.shape[1], -1)
                        embeddings_ = (topic_masks * outputs[1][-1]).sum(dim=1).detach().to('cpu').numpy()
                    embeddings.append(embeddings_)
                    if i % 50 == 0:
                        print(f"{i}/{len(self.loader)}")
            print('Done')
            embeddings = np.concatenate(embeddings, axis=0)
            pickle.dump(embeddings, open(embedding_path, 'wb'))
        else:
            embeddings = pickle.load(open(embedding_path, 'rb'))

        return embeddings

    def plot_embeddings(self, topic_emb=False, dim_reduction='pca'):
        '''
        Plot the document embeddings.
        :param topic_emb: see "calc_embeddings()"
        :param dim_reduction: which dimension reduction method to use. PCA or TSNE or UMAP
        :return:
        '''
        os.makedirs('embeddings', exist_ok=True)
        print('Plotting....')
        print('Reducing dimension....')
        if not topic_emb:
            embedding_path = f'embeddings/embeddings_{dim_reduction}_unfinetuned={self.args.unfinetuned}.pkl'
        else:
            embedding_path = f'embeddings/topic_embeddings_{dim_reduction}_unfinetuned={self.args.unfinetuned}.pkl'
        if self.args.shuffle:
            embedding_path = embedding_path[:-4] + '_shuffle.pkl'
        if not os.path.exists(embedding_path):
            embeddings = self.calc_embeddings(topic_emb)
            if dim_reduction == 'pca':
                from sklearn.decomposition import PCA
                embeddings2 = PCA(n_components=2).fit_transform(embeddings)
            elif dim_reduction == 'tsne':
                from sklearn.manifold import TSNE
                embeddings2 = TSNE(n_components=2).fit_transform(embeddings)
            else:
                from umap import UMAP
                embeddings2 = UMAP(n_neighbors=15, n_components=2, min_dist=0, metric='cosine').fit_transform(embeddings)
            pickle.dump(embeddings2, open(embedding_path, 'wb'))
        else:
            embeddings2 = pickle.load(open(embedding_path, 'rb'))
        print('Done')
        data = pd.DataFrame(embeddings2, columns=['x', 'y'])
        data['labels'] = self.labels
        df_doc_topic = pd.read_csv(os.path.join(self.args.data_path, 'df_doc_topic.csv'))
        df_doc_topic = df_doc_topic.sort_values(by=['prob'], ascending=False).drop_duplicates(subset='idx_doc',
                                                                                              keep='first')

        data['cluster_labels'] = -1
        data['cluster_labels'][df_doc_topic['idx_doc'].tolist()] = df_doc_topic['idx_topic'].tolist()
        # only plot the documents in the 10 labeled topics
        data = data[data['cluster_labels'].isin([1, 2, 8, 9, 10, 11, 12, 27, 30, 33])]

        import matplotlib.pyplot as plt
        clustered = data[data['cluster_labels'] != -1]
        clustered1 = clustered[clustered['labels'] == 0][:200]
        clustered2 = clustered[clustered['labels'] == 1][:200]

        from matplotlib.backends.backend_pdf import PdfPages
        os.makedirs('fig', exist_ok=True)
        if not topic_emb:
            fig_name = f'fig/embeddings_{dim_reduction}_unfinetuned={self.args.unfinetuned}.pdf'
            print(fig_name)
        else:
            fig_name = f'fig/topic_embeddings_{dim_reduction}_unfinetuned={self.args.unfinetuned}.pdf'
            print(fig_name)

        with PdfPages(fig_name) as pdf:
            _, _ = plt.subplots(figsize=(5, 5))
            plt.scatter(clustered1.x, clustered1.y, c=clustered1['cluster_labels'], marker='o', s=30, cmap='hsv_r', alpha=0.2,
                        label='liberal')
            plt.scatter(clustered2.x, clustered2.y, c=clustered2['cluster_labels'], marker='x', s=30, cmap='hsv_r', alpha=0.5,
                        label='conservative')

            plt.xlabel('dim_1', fontsize=12)
            plt.ylabel('dim_2', fontsize=12)
            plt.legend(fontsize=12)
            pdf.savefig()

    def get_polarization(self):
        '''
        calculate the polarization score for each topic and save the ranking
        '''

        def select_docs(df_doc_topic, topic_idx, source, month, max_docs=10, min_docs=2):
            '''
            output the top-n documents from each source for each topic
            '''
            if not isinstance(source, list):
                source = [source]
            if not isinstance(month, list):
                month = [month]

            df = df_doc_topic[(df_doc_topic['idx_topic'] == topic_idx) &
                              (df_doc_topic['month'].isin(month)) &
                              (df_doc_topic['source'].isin(source))].sort_values(by=['prob'],
                                                                                 ascending=False).head(max_docs)
            if df.shape[0] >= min_docs:
                return df['idx_doc'].tolist(), df['prob'].tolist()
            return [], []

        def calc_corpus_embedding(text_embeddings, text_probs):
            # calculate corpus-contextualized document embeddings
            # text_probs: the probabilities of a doc associated with the topic
            if len(text_embeddings) != 0:
                text_probs = np.array(text_probs)
                text_probs /= text_probs.mean()
                text_probs = text_probs.reshape(-1, 1)
                return (text_probs * text_embeddings).mean(axis=0)
            else:
                return np.zeros(768)

        topics = pickle.load(open(os.path.join(self.args.data_path, 'topics.pkl'), 'rb'))
        topic_stems = [[each[0] for each in each1[1]] for each1 in topics]

        if args.polarization in ['emb', 'emb_pairwise']:
            doc_embeddings = self.calc_embeddings(True)
        elif args.polarization == 'emb_doc':
            doc_embeddings = self.calc_embeddings()
        else:
            doc_embeddings = None

        df_doc_topic = pd.read_csv(os.path.join(self.args.data_path, 'df_doc_topic.csv'))

        ### the annotations of the document leanings for documents in the 10 labels topics
        doc_idx2label = pickle.load(open(os.path.join(self.args.data_path, 'doc_idx2label.pkl'), 'rb'))

        topic_ranks = pickle.load(open(os.path.join(self.args.data_path, 'topic_ranks.pkl'), 'rb'))
        # topic_idxes = [each[0] for each in topic_ranks]
        topic_idxes = [1, 2, 8, 9, 10, 11, 12, 27, 30, 33]
        months = sorted(df_doc_topic['month'].unique().tolist())

        if args.polarization in ['emb', 'emb_pairwise', 'emb_doc']:
            corpus, id2word = None, None
        else:
            corpus, id2word = pickle.load(open(os.path.join(self.args.data_path, 'corpus_lo.pkl'), 'rb'))

        data = []
        from sklearn.metrics.pairwise import cosine_similarity
        for topic_idx in topic_idxes:
            print(f"{'*' * 10}Topic: {topic_idx}{'*' * 10}")
            row = [','.join(topic_stems[topic_idx]), f'topic_{topic_idx}']

            months_ = [months]
            for month in months_:
                idxes_docs1, text_probs1 = select_docs(df_doc_topic, topic_idx, self.args.source1, month,
                                                       self.args.max_docs, self.args.min_docs)
                idxes_docs2, text_probs2 = select_docs(df_doc_topic, topic_idx, self.args.source2, month,
                                                       self.args.max_docs, self.args.min_docs)
                min_len = min(len(idxes_docs1), len(idxes_docs2))
                print(f"month:{month}/{max(months)}, n_docs:{min_len}")
                idxes_docs1_ = idxes_docs1[:min_len]
                idxes_docs2_ = idxes_docs2[:min_len]
                text_probs1 = np.array(text_probs1[:min_len])
                text_probs2 = np.array(text_probs2[:min_len])
                text_probs1 /= text_probs1.mean()
                text_probs2 /= text_probs2.mean()

                if self.args.polarization in ['emb', 'emb_pairwise', 'emb_doc']:

                    if args.polarization in ['emb', 'emb_doc']:
                        emb1 = calc_corpus_embedding(doc_embeddings[idxes_docs1_], text_probs1)
                        emb2 = calc_corpus_embedding(doc_embeddings[idxes_docs2_], text_probs2)
                        cos_sim = cosine_similarity([emb1], [emb2])[0][0]
                    else:
                        embs1_ = doc_embeddings[idxes_docs1_]
                        embs2_ = doc_embeddings[idxes_docs2_]
                        if embs1_.sum() != 0 and embs2_.sum() != 0:
                            pairwise_cossim = cosine_similarity(embs1_, embs2_)
                            weight_mat = np.matmul(np.array(text_probs1).reshape(-1, 1),
                                                   np.array(text_probs2).reshape(1, -1))
                            # weight_mat = np.ones((min_len, min_len))
                            weight_mat = weight_mat / weight_mat.mean()
                            cos_sim = (pairwise_cossim * weight_mat).mean()
                        else:
                            cos_sim = float('nan')
                    pola_score = 0.5 * (-cos_sim + 1)

                elif self.args.polarization == 'lo':
                    corpus1 = pd.Series(corpus)[idxes_docs1]
                    corpus2 = pd.Series(corpus)[idxes_docs2]
                    pola_score, pol_score_random, n_articles = get_leaveout_score(corpus1, corpus2, id2word,
                                                                                  min_docs=args.min_docs,
                                                                                  max_docs=args.max_docs)
                    # pola_score = 1 - 2 * pola_score
                else:  # ground true
                    annotations1 = [doc_idx2label[each] for each in idxes_docs1]
                    annotations2 = [doc_idx2label[each] for each in idxes_docs2]
                    label2indexes1 = {0: [], 1: [], -1: []}
                    label2indexes2 = {0: [], 1: [], -1: []}
                    for i, anno in enumerate(annotations1):
                        label2indexes1[anno].append(i)
                    for i, anno in enumerate(annotations2):
                        label2indexes2[anno].append(i)

                    x = 0
                    for i in range(len(label2indexes1[1])):
                        index = label2indexes1[1][i]
                        prob = text_probs1[index]
                        x += prob
                    y = 0
                    for i in range(len(label2indexes1[0])):
                        index = label2indexes1[0][i]
                        prob = text_probs1[index]
                        y += prob
                    score1 = (x-y)/len(annotations1)

                    x = 0
                    for i in range(len(label2indexes2[1])):
                        index = label2indexes2[1][i]
                        prob = text_probs2[index]
                        x += prob
                    y = 0
                    for i in range(len(label2indexes2[0])):
                        index = label2indexes2[0][i]
                        prob = text_probs2[index]
                        y += prob
                    score2 = (x - y) / len(annotations2)

                    pola_score = np.abs(score1 - score2) / 2

                # pola_score = float('nan') if pola_score == 0 else pola_score
                row.append(pola_score)

            data.append([row[1], row[2], row[0]])
            print()

        os.makedirs('results', exist_ok=True)
        file_name = f"{args.source1}_{args.source2}{'_unfinetuned' if args.unfinetuned else ''}_{args.polarization}" \
                    f"_{args.max_docs}_{args.min_docs}.csv"
        if self.args.shuffle:
            file_name = file_name[:-5] + '_shuffle.csv'
        data.sort(key=lambda x: x[1], reverse=True)
        pd.DataFrame(data, columns=['topic_idx', 'pola'] + ['topic_words']).to_csv(
            f"results/{file_name}",
            index=False)
        print('Done')
        print(file_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../01_topic_modeling/data')

    # training BERT model
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--unfinetuned', type=int, choices=(0, 1), default=0,
                        help='whether to finetune the language model or not')
    parser.add_argument('--gpu', type=str, default='', help='which gpus to use, starting from 0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_train', type=int, choices=(0,1), default=1)

    parser.add_argument('--shuffle', type=int, choices=(0, 1), default=0,
                        help='whether to shuffle the partisanship labels or not')

    # plotting
    parser.add_argument('--plotting', type=int, choices=(0, 1), default=1, help='whether to plot the embeddings or not')
    parser.add_argument('--dim_reduction', type=str, choices=('pca', 'tsne', 'umap'), default='tsne', help='which ')

    # calculating polarization
    parser.add_argument('--polarization', type=str,
                        choices=('emb', # pacte
                                 'lo',  # leaveout estimator
                                 'emb_pairwise',   # a baseline, never mind
                                 'gt',  # ground truth from annotations
                                 'emb_doc'  # pacte, but the document embedding is the holistic CLS embedding
                                ),
                        default='emb',
                        help='the method to use to calculate the polarization scores')
    parser.add_argument('--source1', nargs='+', default=['cnn', 'huff', 'nyt'], help='the left sources')
    parser.add_argument('--source2', nargs='+', default=['fox', 'breit', 'nyp'], help='the right sources')
    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--max_docs', type=int, default=10, help='max # of documents for a topic from each source')
    parser.add_argument('--min_docs', type=int, default=10, help='min # of documents for a topic from each source')

    args = parser.parse_args()
    print(args)
    if args.polarization in ['lo', 'gt']:
        args.unfinetuned = 0
        args.init_train = 0
        # args.plotting = 0

    args.unfinetuned = {0: False, 1: True}[args.unfinetuned]
    args.init_train = {0: False, 1: True}[args.init_train]
    args.plotting = {0: False, 1: True}[args.plotting]
    args.shuffle = {0: False, 1: True}[args.shuffle]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = Engine(args)
    if args.init_train:
        engine.train()
    engine.get_polarization()
    if args.plotting:
        engine.plot_embeddings(dim_reduction='pca')
        engine.plot_embeddings(dim_reduction='tsne')
        engine.plot_embeddings(topic_emb=True, dim_reduction='pca')
        engine.plot_embeddings(topic_emb=True, dim_reduction='tsne')

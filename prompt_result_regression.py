import random
from argparse import ArgumentParser, Namespace
import pandas as pd
from tqdm import tqdm
from utils.data import pad_and_concat
import os
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
import statsmodels.api as sm
import json
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
import plotly.express as px


# utils
class GetFullDF:
    def __init__(self, directory):
        self.directory = directory
        self.df = None
    def __call__(self):
        if self.df is None:
            df = pd.concat([pd.read_csv(os.path.join(self.directory, file)) for file in os.listdir(self.directory)])
            df = df[~((df['split'] == 'train') &
                      ((df.dev_slice_args == '[800, 900]') |
                       (df.dev_slice_args == '[900, 1000]')))]
            df = df.drop_duplicates(['k', 'task', 'prompt_seed', 'train_slice_args', 'dev_slice_args', 'subsample_dev',
                                     'checkpoint', 'split'])
            df = df.reset_index()
            df['sampling'] = 'within_task_random'
            df['method'] = 'in_context'
            df['descriptor'] = df[['checkpoint', 'sampling', 'method']].apply(tuple, axis=1)
            df['logprob'] = -df['loss']
            df['prob'] = np.exp(df['logprob'])
            def get_dev_size(r):
                if r.subsample_dev == r.subsample_dev:
                    return r.subsample_dev
                else:
                    x1, x2 = eval(r.dev_slice_args)
                    return x2 - x1
            df['evaluated_on_size'] = df.apply(get_dev_size, axis=1)
            df['normalizedlogprob'] = -df['normalized_loss']
            df['normalizedprob'] = np.exp(df['normalizedlogprob'])
            for k in ['logprob', 'normalizedlogprob', 'loss', 'normalized_loss', 'acc', 'f1']:
                df[k + '_normalized'] = df[k]
                for x in set(df.dev_slice_args):
                    values = df[k][df.dev_slice_args == x]
                    df[k + '_normalized'][df.dev_slice_args == x] = values - values.mean()
            df['prob_normalized'] = np.exp(df['logprob_normalized'])
            df['normalizedprob_normalized'] = np.exp(df['normalizedlogprob_normalized'])
            self.df = df
        return self.df
def evaluate_prediction_df(df, key_predicted, key_real, split, table):
    corr = df.regression_predictions.corr(df[key])
    new_df = pd.DataFrame(
        {'predicted %s' % key_predicted: df['regression_predictions'],
         'real %s' % key_real: df[key_real],
         'val data used': df['dev_slice_args']})
    fig = px.scatter(new_df, x='predicted %s' % key_predicted, y='real %s' % key_real, color='val data used')
    path_to_plotly_html = "./plotly_figure.html"
    fig.write_html(path_to_plotly_html, auto_play=False)
    table.add_data(
        split, key_predicted, key_real, len(new_df), corr, wandb.Html(path_to_plotly_html))
# def evaluate_prediction_dfs(train_df, val_df, key, outdir, model):
#     print('train corr: %f, val corr: %f' % (
#         train_df.regression_predictions.corr(train_df[key]),
#         val_df.regression_predictions.corr(val_df[key])))
#     chart = sns.scatterplot(data=train_df, x=key, y='regression_predictions', hue='dev_slice_args')
#     plt.show()
#     chart = sns.scatterplot(data=val_df, x=key, y='regression_predictions', hue='dev_slice_args')
#     plt.show()
def get_breakdown(df):
    ckpts = sorted(list(set(df.checkpoint)))
    tasks = sorted(list(set(df.task)))
    samplings = sorted(list(set(df.sampling)))
    methods = sorted(list(set(df.method)))
    splits = list(set(df.split))
#     evalsizes = list(set(df.evaluated_on_size))
    devslices = sorted(list(set(df.dev_slice_args)))
    breakdown_df = pd.DataFrame({
        (task, devslice): {
            (ch, sampling, method, split): len(df[
                (df.task == task) & (df.checkpoint == ch) &
                (df.sampling == sampling) & (df.method == method) &
                (df['split'] == split) & (df.dev_slice_args == devslice)])
            for ch in ckpts for sampling in samplings for method in methods for split in splits
        } for task in tasks for devslice in devslices}).transpose()
    print(len(breakdown_df))
    return breakdown_df


# Data
def get_prompt(train_samples):
    train_samples = eval(train_samples)
    return '\n\n'.join(['input: %s\noutput: %s\n' % (dp['input'], dp['output']) for dp in train_samples])
class PromptPerformanceDataset(Dataset):
    def __init__(self, df, key):
        self.df = df
        self.key = key
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return dict(
            prompt=get_prompt(row.train_samples),
            label=row[self.key],
        )
class PromptsPerformanceData(pl.LightningDataModule):
    def __init__(self, get_df_func, key='normalizedlogprob', train_percent=.8, val_percent=.1, seed=0, setting=None,
                 **dataloader_kwargs):
        super().__init__()
        self.get_df_func = get_df_func
        self.dataloader_kwargs = dataloader_kwargs
        self._train, self._val, self._test = None, None, None
        self.train_percent, self.val_percent, self.seed = train_percent, val_percent, seed
        self.setting = setting
        self.key = key
    def setup(self, stage=None):
        df = self.get_df_func()
        df = df[df.k != 0]
        if self.setting == 'no_example_overlap':
            self._train, self._val, self._test = [
                PromptPerformanceDataset(df[df['split'] == split], self.key)
                for split in ['train', 'val', 'test']]
        elif self.setting == 'no_dev_overlap':
#             df = df[df.train_data_split == 'train']
            df = df[df['split'] == 'train']
#             length = len(df)
            dev_slices = sorted(list(set(df.dev_slice_args)))
            length = len(dev_slices)
            train_length = round(length * self.train_percent)
            val_length = round(length * self.val_percent)
#             lengths = [train_length, val_length, length - train_length - val_length]
#             rows = [r for i, r in df.iterrows()]
#             splits = random_split(rows, lengths, generator=torch.Generator().manual_seed(self.seed))
#             self._train, self._val, self._test = [
#                 PromptPerformanceDataset(pd.DataFrame(list(x)), self.key) for x in splits]
            slice_sets = [
                dev_slices[:train_length],
                dev_slices[train_length:train_length + val_length],
                dev_slices[train_length + val_length:]]
            self._train, self._val, self._test = [
                PromptPerformanceDataset(df[df.apply(lambda r: r.dev_slice_args in slice_set, axis=1)], self.key)
                for slice_set in slice_sets]
        elif self.setting is None:
#             df = df[df.train_data_split == 'train']
            df = df[df['split'] == 'train']
            length = len(df)
            train_length = round(length * self.train_percent)
            val_length = round(length * self.val_percent)
            lengths = [train_length, val_length, length - train_length - val_length]
            rows = [r for i, r in df.iterrows()]
            splits = random_split(rows, lengths, generator=torch.Generator().manual_seed(self.seed))
            self._train, self._val, self._test = [
                PromptPerformanceDataset(pd.DataFrame(list(x)), self.key) for x in splits]
        else:
            raise NotImplementedError
    def train_dataloader(self):
        return DataLoader(self._train, **self.dataloader_kwargs, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self._val, **self.dataloader_kwargs)
    def test_dataloader(self):
        return DataLoader(self._test, **self.dataloader_kwargs)


# Linear Regression on Per-example Indicator Variables
def get_features(indices):
    x = np.zeros(500)
    x[np.array(indices)] = 1
    return x
def getxy(df, key):
    X = np.stack(df.apply(lambda r: get_features(eval(r.train_indices)), axis=1))
    y = np.array(df[key])
    return X, y
def setup_and_train_linear_regression(get_full_df, task, checkpoint, key, save_coefs=False, setting=None):
    def get_df():
        df = get_full_df()
        return df[(df.task == task) & (df.checkpoint == checkpoint)]
    dm = PromptsPerformanceData(get_df, setting=setting)
    dm.setup()
    X_train, y_train = getxy(dm._train.df, key)
    print('train shapes: X - %s, y - %s' % (str(X_train.shape), str(y_train.shape)))
    X_val, y_val = getxy(dm._val.df, key)
    print('val shapes: X - %s, y - %s' % (str(X_val.shape), str(y_val.shape)))
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
#     print(model.summary())
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    if save_coefs:
        outdir = wandb.run.dir
        if not os.path.exists(os.path.join(outdir, 'regcoef')):
            os.mkdir(os.path.join(outdir, 'regcoef'))
        if not os.path.exists(os.path.join(outdir, 'regcoef', checkpoint.replace('/', '-'))):
            os.mkdir(os.path.join(outdir, 'regcoef', checkpoint.replace('/', '-')))
#         np.save(os.path.join(outdir, 'regcoef', checkpoint.replace('/', '-'), '%s.npy' % task), model.coef_)
        np.save(os.path.join(outdir, 'regcoef', checkpoint.replace('/', '-'), '%s.npy' % task), model.params[1:])
        if not os.path.exists(os.path.join(outdir, 'negregcoef')):
            os.mkdir(os.path.join(outdir, 'negregcoef'))
        if not os.path.exists(os.path.join(outdir, 'negregcoef', checkpoint.replace('/', '-'))):
            os.mkdir(os.path.join(outdir, 'negregcoef', checkpoint.replace('/', '-')))
        np.save(os.path.join(outdir, 'negregcoef', checkpoint.replace('/', '-'), '%s.npy' % task), -model.params[1:])
    return dm, model
def get_prediction_dfs_linear_regression(dm, model, key):
    X_train, y_train = getxy(dm._train.df, key)
#     y_train_pred = model.predict(X_train)
    y_train_pred = model.predict(sm.add_constant(X_train))
    train_df = dm._train.df.copy()
    train_df['regression_predictions'] = y_train_pred
    X_val, y_val = getxy(dm._val.df, key)
#     y_val_pred = model.predict(X_val)
    y_val_pred = model.predict(sm.add_constant(X_val))
    val_df = dm._val.df.copy()
    val_df['regression_predictions'] = y_val_pred
#     print('train regression score: %s, val regression score: %s' % (model.score(X_train, y_train), model.score(X_val, y_val)))
    return train_df, val_df


# Transformer Regression
class StringRegressionModel(pl.LightningModule):
    def __init__(self, lr=1e-4, from_pretrained='roberta-base'):
        super().__init__()
        self.model = AutoModel.from_pretrained(from_pretrained)
        self.model.parameters()
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
#         self.linear = nn.Linear(1024, 1)
        self.loss = nn.MSELoss()
        self.lr = lr
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, batch):
        out = self.model(**batch['text'])
        predictions = self.linear(out[1]).squeeze(-1)
        return predictions
    def shared_step(self, batch, step_type):
        predictions = self(batch)
        loss = self.loss(predictions, batch['label'])
        corr = pd.Series(predictions.detach().cpu().numpy()).corr(pd.Series(batch['label'].cpu().numpy()))
        self.log('loss/%s' % step_type, loss.item())
        self.log('corr/%s' % step_type, corr)
        return loss
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')
    def predict_step(self, batch, batch_idx):
        return self(batch).cpu().numpy()
def part_requires_grad(m, rg):
    for p in m.parameters():
        p.requires_grad = rg
def freeze_roberta(model):
    part_requires_grad(model.embeddings, True)
    for i, m in enumerate(model.encoder.layer):
        if i < len(model.encoder.layer) - 2:
            part_requires_grad(m, False)
        else:
            part_requires_grad(m, True)
    part_requires_grad(model.pooler, True)
class CollateFn:
    def __init__(self, from_pretrained='roberta-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
    def __call__(self, instances):
        return dict(
            text=self.tokenizer([self.tokenizer.cls_token + ' ' + i['prompt'] for i in instances], return_tensors='pt', padding=True, truncation=True),
            label=torch.tensor([i['label'] for i in instances], dtype=torch.float32)
        )
def setup_and_train_roberta_regression(get_full_df, task, checkpoint, key,
                                       epochs=10, no_training=False, batch_size=64, lr=1e-4, seed=0, setting=None,
                                       from_pretrained='roberta-base'):
    def get_df():
        df = get_full_df()
        return df[(df.task == task) & (df.checkpoint == checkpoint)]
    dm = PromptsPerformanceData(get_df, key=key, batch_size=batch_size,
                                collate_fn=CollateFn(from_pretrained=from_pretrained), seed=seed,
                                setting=setting)
    dm.setup()
    trainer = pl.Trainer(
        gpus=1, max_epochs=epochs, log_every_n_steps=10, logger=WandbLogger(),
        callbacks=[
            # EarlyStopping(monitor="loss/val", mode="min", patience=20),
            ModelCheckpoint(dirpath=get_checkpooint_dir(), save_top_k=3, save_last=True,
                            monitor="corr/val", mode='max')
        ],
    )
    model = StringRegressionModel(lr=lr, from_pretrained=from_pretrained)
    freeze_roberta(model.model)
    if not no_training:
        trainer.fit(model, datamodule=dm,
            **(dict(ckpt_path=os.path.join(get_checkpooint_dir(), 'last.ckpt'))
               if wandb.run.resumed else {}))
        best_model_path = trainer.checkpoint_callback.best_model_path
    else:
        assert wandb.run.resumed
        last_checkpoint = torch.load(os.path.join(get_checkpooint_dir(), 'last.ckpt'))
        best_model_path = None
        for k, v in last_checkpoint['callbacks'].items():
            if k.startswith('ModelCheckpoint'):
                best_model_path = v['best_model_path']
                break
    print('restoring the best checkpoint: %s' % best_model_path)
    best_weights = torch.load(best_model_path)
    model.load_state_dict(best_weights['state_dict'])
    return trainer, dm, model
def get_prediction_dfs_roberta_regression(trainer, dm, model):
    train_reg_preds = np.concatenate(trainer.predict(model, dm.train_dataloader()))
    train_df = dm._train.df.copy()
    train_df['regression_predictions'] = train_reg_preds
    val_reg_preds = np.concatenate(trainer.predict(model, dm.val_dataloader()))
    val_df = dm._val.df.copy()
    val_df['regression_predictions'] = val_reg_preds
    return train_df, val_df


# Pairwise Ranking of Prompts
class PromptsPerformanceDataByTestSubset(PromptsPerformanceData):
    def setup(self, stage=None):
        super().setup(stage=stage)
        dev_slice_arg_set = set(self._train.df.dev_slice_args)
        self._train = [
            PromptPerformanceDataset(self._train.df[self._train.df.dev_slice_args == x], self.key)
            for x in dev_slice_arg_set]
        dev_slice_arg_set = set(self._val.df.dev_slice_args)
        self._val = [
            PromptPerformanceDataset(self._val.df[self._val.df.dev_slice_args == x], self.key)
            for x in dev_slice_arg_set]
        dev_slice_arg_set = set(self._test.df.dev_slice_args)
        self._test = [
            PromptPerformanceDataset(self._test.df[self._test.df.dev_slice_args == x], self.key)
            for x in dev_slice_arg_set]
    def train_dataloader(self):
        return [DataLoader(d, **self.dataloader_kwargs, shuffle=True) for d in self._train]
    def val_dataloader(self):
        return [DataLoader(d, **self.dataloader_kwargs) for d in self._val]
    def test_dataloader(self):
        return [DataLoader(d, **self.dataloader_kwargs) for d in self._test]
class StringPairRankingModel(pl.LightningModule):
    def __init__(self, lr=1e-4, from_pretrained='roberta-base', sample_one_batch=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(from_pretrained)
        self.vec1 = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            # nn.Tanh(),
        )
        self.vec2 = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            # nn.Tanh(),
        )
        self.lr = lr
        self.sample_one_batch = sample_one_batch
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, batch):
        out = self.model(**batch['text'])
        vecs1 = self.vec1(out[1])
        vecs2 = self.vec2(out[1])
        return vecs1, vecs2
    def shared_step(self, batch, step_type):
        batches = [batch] if not isinstance(batch, list) else batch
        if self.sample_one_batch:
            batches = random.choices(batches, k=1)
        lengths, input_ids, attention_mask = [], [], []
        for batch in batches:
            input_ids.append(batch['text']['input_ids'])
            attention_mask.append(batch['text']['attention_mask'])
            lengths.append(batch['text']['input_ids'].shape[0])
        input_ids = pad_and_concat(input_ids)
        input_ids = input_ids.reshape(-1, *input_ids.shape[2:])
        attention_mask = pad_and_concat(attention_mask)
        attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:])
        allvecs1, allvecs2 = self({'text': {'input_ids': input_ids, 'attention_mask': attention_mask}})
        losses = []
        accs = []
#         for batch in batches:
#             vecs1, vecs2 = self(batch)
        offset = 0
        for length in lengths:
            vecs1, vecs2 = allvecs1[offset:offset+length], allvecs2[offset:offset+length]
            offset += length
            predictions = torch.mm(vecs1, vecs2.transpose(0, 1))
            predictions = torch.sigmoid(predictions)
            loss = (- torch.log(predictions[batch['label'] == 1]).sum()
                    - torch.log(1 - predictions[batch['label'] == -1]).sum()) \
                   / (batch['label'] != 0).sum()
            losses.append(loss)
            acc = (((predictions > .5) * 2 - 1) == batch['label']).sum() / (batch['label'] != 0).sum()
#             self.log('acc/%s' % step_type, acc.item())
            accs.append(acc)
#             return loss
        losses = torch.stack(losses)
        accs = torch.stack(accs)
        self.log('loss/%s' % step_type, losses.mean().item())
        self.log('acc/%s' % step_type, accs.mean().item())
        return losses.mean()
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')
    def predict_step(self, batch, batch_idx):
        vecs1, vecs2 = self(batch)
        return vecs1.cpu().numpy(), vecs2.cpu().numpy()
class PairRankingCollateFn:
    def __init__(self, from_pretrained='roberta-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
    def __call__(self, instances):
        batch = dict(
            text=self.tokenizer([self.tokenizer.cls_token + ' ' + i['prompt'] for i in instances], return_tensors='pt', padding=True, truncation=True),
        )
        if 'label' in instances[0].keys():
            labels = []
            for i, i1 in enumerate(instances):
                labels.append([])
                for j, i2 in enumerate(instances):
                    labels[-1].append(0 if i == j or i1['label'] == i2['label'] else 1 if i1['label'] > i2['label'] else -1)
            batch['label'] = torch.tensor(labels, dtype=torch.float32)
        return batch
def setup_and_train_roberta_pair_ranking(get_full_df, task, checkpoint, key,
                                         epochs=10, batch_size=64, lr=1e-5, seed=0, setting=None,
                                         from_pretrained='roberta-base', no_training=False):
    def get_df():
        df = get_full_df()
        return df[(df.task == task) & (df.checkpoint == checkpoint)]
#     dm = PromptsPerformanceData(get_df, key=key, batch_size=batch_size, collate_fn=PairRankingCollateFn(),
#                                 seed=seed,
#                                 setting=setting)
    dm = PromptsPerformanceDataByTestSubset(
        get_df, key=key, batch_size=batch_size, collate_fn=PairRankingCollateFn(from_pretrained=from_pretrained),
        seed=seed,
        setting=setting)
    dm.setup()
    trainer = pl.Trainer(
        gpus=1, max_epochs=epochs, log_every_n_steps=10, logger=WandbLogger(),
        callbacks=[
            # EarlyStopping(monitor="loss/val", mode="min", patience=100),
            ModelCheckpoint(dirpath=get_checkpooint_dir(), save_top_k=3, save_last=True,
                            monitor="acc/val", mode='max')
        ],
    )
    model = StringPairRankingModel(lr=lr, from_pretrained=from_pretrained)
    freeze_roberta(model.model)
    if not no_training:
        trainer.fit(model, datamodule=dm,
            **(dict(ckpt_path=os.path.join(get_checkpooint_dir(), 'last.ckpt'))
               if wandb.run.resumed else {}))
        best_model_path = trainer.checkpoint_callback.best_model_path
    else:
        assert wandb.run.resumed
        last_checkpoint = torch.load(os.path.join(get_checkpooint_dir(), 'last.ckpt'))
        best_model_path = None
        for k, v in last_checkpoint['callbacks'].items():
            if k.startswith('ModelCheckpoint'):
                best_model_path = v['best_model_path']
                break
    print('restoring the best checkpoint: %s' % best_model_path)
    best_weights = torch.load(best_model_path)
    model.load_state_dict(best_weights['state_dict'])
    return trainer, dm, model
class RowKey:
    def __init__(self, vec1, vec2, pbar):
        self.vec1 = vec1
        self.vec2 = vec2
        self.pbar = pbar
    def get_comparison_score(self, other):
        x = (self.vec1[None, :] @ other.vec2[:, None])[0, 0]
        x = 1/(1 + np.exp(-x))
        self.pbar.update(1)
        return x
    def __gt__(self, other):
        return self.get_comparison_score(other) > .5
    def __lt__(self, other):
        return not (self > other)
    def __ge__(self, other):
        return self > other
    def __le__(self, other):
        return self < other
def get_roberta_pair_ranking(trainer, dm, model, split='val'):
    if split == 'val':
        allrows = [dataset.df.copy() for dataset in dm._val]
        alldls = dm.val_dataloader()
    elif split == 'train':
        allrows = [dataset.df.copy() for dataset in dm._train]
        alldls = dm.train_dataloader()
    else:
        raise NotImplementedError
    for rows, dl in zip(allrows, alldls):
        print('sorting %i rows' % len(rows))
        with torch.no_grad():
            predictions = trainer.predict(model, dl)
            predictions = [(vec1, vec2) for vecs1, vecs2 in predictions for vec1, vec2 in zip(vecs1, vecs2)]
            with tqdm() as pbar:
                original_spots = [
                    x[0] for x in sorted(
                        [(i, vec1, vec2)
                         for i, (vec1, vec2) in enumerate(predictions)],
                        key=lambda x: RowKey(x[1], x[2], pbar))
                ]
        rankings = [-1] * len(rows)
        for i, original_spot in enumerate(original_spots):
            rankings[original_spot] = i
        rows['regression_predictions'] = rankings
    return pd.concat(allrows)


def get_checkpooint_dir():
    return os.path.join('/'.join(wandb.run.dir.split('/')[:-3]), 'checkpoints', wandb.run.id)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--indir')
    parser.add_argument('--outdir')
    parser.add_argument('--mode')
    parser.add_argument('--task', default='commonsense_qa')
    parser.add_argument('--setting', default=None)
    parser.add_argument('--from_pretrained', default='roberta-base')
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--no_training', default=False, action='store_true')
    args = parser.parse_args()
    wandb.init(config=args, project="prompt_regression", dir=args.outdir,
               **(dict(id=args.run_id, resume="must") if args.run_id is not None else {}))
    args = Namespace(**wandb.config)
    get_full_df = GetFullDF(args.indir)
    print(get_breakdown(get_full_df()))
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(os.path.join(args.outdir, 'checkpoints')):
        os.mkdir(os.path.join(args.outdir, 'checkpoints'))
    if args.mode == 'indicator_regression':
        # Linear Regression on Per-example Indicator Variables
        table = wandb.Table(columns=['split', 'predicted', 'real', 'len', 'corr', 'figure'])
        key = 'acc'
        dm, model = setup_and_train_linear_regression(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key, save_coefs=False,
            setting=args.setting)
        train_df, val_df = get_prediction_dfs_linear_regression(dm, model, key)
        evaluate_prediction_df(train_df, key, key, 'train', table)
        evaluate_prediction_df(val_df, key, key, 'val', table)
        key = 'acc_normalized'
        dm, model = setup_and_train_linear_regression(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key, save_coefs=True,
            setting=args.setting)
        train_df, val_df = get_prediction_dfs_linear_regression(dm, model, key)
        evaluate_prediction_df(train_df, key, key, 'train', table)
        evaluate_prediction_df(val_df, key, key, 'val', table)
        key = 'normalizedprob_normalized'
        dm, model = setup_and_train_linear_regression(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key, save_coefs=False,
            setting=args.setting)
        train_df, val_df = get_prediction_dfs_linear_regression(dm, model, key)
        evaluate_prediction_df(train_df, key, key, 'train', table)
        evaluate_prediction_df(val_df, key, key, 'val', table)
        wandb.log({'table': table})
    elif args.mode == 'transformer_regression':
        key = 'acc_normalized'
        trainer, dm, model = setup_and_train_roberta_regression(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key,
            epochs=300, no_training=args.no_training,
            batch_size=48, lr=3e-6,
            setting=args.setting)
        train_df, val_df = get_prediction_dfs_roberta_regression(trainer, dm, model)
        table = wandb.Table(columns=['split', 'predicted', 'real', 'len', 'corr', 'figure'])
        evaluate_prediction_df(train_df, key, key, 'train', table)
        evaluate_prediction_df(val_df, key, key, 'val', table)
        evaluate_prediction_df(train_df, key, 'f1_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'f1_normalized', 'val', table)
        wandb.log({'table': table})
    elif args.mode == 'pairwise_comparator_ranking':
        key = 'acc_normalized'
        trainer, dm, model = setup_and_train_roberta_pair_ranking(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key,
            epochs=300, no_training=args.no_training,
            batch_size=48, lr=3e-6,
            setting=args.setting)
        val_df = get_roberta_pair_ranking(trainer, dm, model, split='val')
        train_df = get_roberta_pair_ranking(trainer, dm, model, split='train')
        table = wandb.Table(columns=['split', 'predicted', 'real', 'len', 'corr', 'figure'])
        evaluate_prediction_df(train_df, key, key, 'train', table)
        evaluate_prediction_df(val_df, key, key, 'val', table)
        evaluate_prediction_df(train_df, key, 'f1_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'f1_normalized', 'val', table)
        wandb.log({'table': table})
    else:
        raise NotImplementedError
    wandb.finish()

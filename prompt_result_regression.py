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
    def __init__(self, directory, tasks=None):
        self.directory = directory
        self.tasks = tasks
        self.df = None
    def __call__(self):
        if self.df is None:
            if self.tasks is not None:
                df = pd.concat([pd.read_csv(os.path.join(self.directory, file)) for file in os.listdir(self.directory)
                                for task in self.tasks
                                if '_'.join(file.split('_')[1:]).startswith(task)])
            else:
                df = pd.concat([pd.read_csv(os.path.join(self.directory, file)) for file in os.listdir(self.directory)])
            df = df[~((df['split'] == 'train') &
                      ((df.dev_slice_args == '[800, 900]') |
                       (df.dev_slice_args == '[900, 1000]')))]
            df_withmodel = df[df.checkpoint==df.checkpoint].drop_duplicates([
                'k', 'task', 'prompt_seed', 'train_slice_args', 'dev_slice_args', 'subsample_dev',
                'checkpoint', 'split'])
            df_withoutmodel = df[df.checkpoint!=df.checkpoint].drop_duplicates([
                'task', 'prompt_seed', 'dev_slice_args', 'subsample_dev', 'split'])
            df = pd.concat([df_withmodel, df_withoutmodel])
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
    corr = df.regression_predictions.corr(df[key_real])
    def get_get_example_text(j):
        def get_example_text(x):
            x = eval(x)[j]
            newinput = x['input'].split(' ')
            newinput = '<br>'.join([' '.join(newinput[i:i+15]) for i in range(0, len(newinput) + 15, 15)])
            return 'Q: %s, A: %s' % (newinput, x['output'])
        return get_example_text
    new_df = pd.DataFrame(
        {'predicted %s' % key_predicted: df['regression_predictions'],
         'real %s' % key_real: df[key_real],
         'val data used': df['dev_slice_args'],
         'name': df['prompt_seed'],
         # 'name': df.apply(
         #     lambda r: '%i (%.2f, %.2f)' % (r.prompt_seed, r['regression_predictions'], r[key_real]), axis=1),
         'example1': df['train_samples'].apply(get_get_example_text(0)),
         'example2': df['train_samples'].apply(get_get_example_text(1)),
         'example3': df['train_samples'].apply(get_get_example_text(2))}
    )
    fig = px.scatter(
        new_df, x='predicted %s' % key_predicted, y='real %s' % key_real, color='val data used',
        hover_name='name', hover_data=['example1', 'example2', 'example3'],
    )
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
    data = {}
    for task in sorted(list(set(df.task))):
        dftemp1 = df[df.task == task]
        for devslice in sorted(list(set(dftemp1.dev_slice_args))):
            dftemp2 = dftemp1[dftemp1.dev_slice_args == devslice]
            data[(task, devslice)] = {}
            for ch in sorted(list(set(dftemp2.checkpoint)), key=lambda x: x if x == x else 'nan'):
                dftemp3 = dftemp2[dftemp2.checkpoint == ch] if ch == ch else dftemp2[dftemp2.checkpoint != dftemp2.checkpoint]
                for sampling in sorted(list(set(dftemp3.sampling)), key=lambda x: x if x == x else 'nan'):
                    dftemp4 = dftemp3[dftemp3.sampling == sampling] if sampling == sampling else dftemp3[dftemp3.sampling != sampling]
                    for method in sorted(list(set(dftemp4.method)), key=lambda x: x if x == x else 'nan'):
                        dftemp5 = dftemp4[dftemp4.method == method] if method == method else dftemp4[dftemp4.method != method]
                        for split in sorted(list(set(dftemp5['split']))):
                            dftemp6 = dftemp5[dftemp5['split'] == split]
                            data[(task, devslice)][(
                                ch if ch == ch else 'nan',
                                sampling if sampling == sampling else 'nan',
                                method if method == method else 'nan',
                                split)] = len(dftemp6)
    breakdown_df = pd.DataFrame(data).transpose()
    print(len(breakdown_df))
    return breakdown_df


# Data
def get_prompt(train_samples, no_labels=False):
    train_samples = eval(train_samples)
    if no_labels:
        return '\n\n'.join([dp['input'] for dp in train_samples])
    return '\n\n'.join(['input: %s\noutput: %s\n' % (dp['input'], dp['output']) for dp in train_samples])
class PromptPerformanceDataset(Dataset):
    def __init__(self, df, key, no_labels=False):
        self.df = df
        self.key = key
        self.no_labels = no_labels
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return dict(
            prompt=get_prompt(row.train_samples, no_labels=self.no_labels),
            label=row[self.key],
        )
class PromptsPerformanceData(pl.LightningDataModule):
    def __init__(self, get_df_func, key='normalizedlogprob', train_percent=.7, val_percent=.2, seed=0, setting=None,
                 no_labels=False,
                 **dataloader_kwargs):
        super().__init__()
        self.get_df_func = get_df_func
        self.dataloader_kwargs = dataloader_kwargs
        self._train, self._val, self._test = None, None, None
        self.train_percent, self.val_percent, self.seed = train_percent, val_percent, seed
        self.setting = setting
        self.key = key
        self.no_labels = no_labels
    def setup(self, stage=None):
        df = self.get_df_func()
        df = df[df.k != 0]
        if self.setting == 'no_example_overlap':
            self._train, self._val, self._test = [
                PromptPerformanceDataset(df[df['split'] == split], self.key, no_labels=self.no_labels)
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
                PromptPerformanceDataset(df[df.apply(lambda r: r.dev_slice_args in slice_set, axis=1)], self.key,
                                         no_labels=self.no_labels)
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
                PromptPerformanceDataset(pd.DataFrame(list(x)), self.key, no_labels=self.no_labels) for x in splits]
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
        outdir = get_regcoef_dir()
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
                                       from_pretrained='roberta-base', no_labels=False):
    def get_df():
        df = get_full_df()
        return df[(df.task == task) & (df.checkpoint == checkpoint)]
    dm = PromptsPerformanceData(get_df, key=key, batch_size=batch_size,
                                collate_fn=CollateFn(from_pretrained=from_pretrained), seed=seed,
                                setting=setting, no_labels=no_labels)
    dm.setup()
    trainer = pl.Trainer(
        gpus=1, max_epochs=epochs, log_every_n_steps=10, logger=WandbLogger(),
        callbacks=[
            # EarlyStopping(monitor="loss/val", mode="min", patience=20),
            ModelCheckpoint(dirpath=get_checkpoint_dir(), save_top_k=3, save_last=True,
                            monitor="corr/val", mode='max')
        ],
    )
    model = StringRegressionModel(lr=lr, from_pretrained=from_pretrained)
    freeze_roberta(model.model)
    if not no_training:
        trainer.fit(model, datamodule=dm,
            **(dict(ckpt_path=os.path.join(get_checkpoint_dir(), 'last.ckpt'))
               if wandb.run.resumed else {}))
        best_model_path = trainer.checkpoint_callback.best_model_path
    else:
        assert wandb.run.resumed
        last_checkpoint = torch.load(os.path.join(get_checkpoint_dir(), 'last.ckpt'))
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
            PromptPerformanceDataset(self._train.df[self._train.df.dev_slice_args == x], self.key,
                                     no_labels=self.no_labels)
            for x in dev_slice_arg_set]
        dev_slice_arg_set = set(self._val.df.dev_slice_args)
        self._val = [
            PromptPerformanceDataset(self._val.df[self._val.df.dev_slice_args == x], self.key,
                                     no_labels=self.no_labels)
            for x in dev_slice_arg_set]
        dev_slice_arg_set = set(self._test.df.dev_slice_args)
        self._test = [
            PromptPerformanceDataset(self._test.df[self._test.df.dev_slice_args == x], self.key,
                                     no_labels=self.no_labels)
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
        if self.sample_one_batch and step_type == 'train':
            batches = random.choices(batches, k=1)
        new_batches, lengths, input_ids, attention_mask = [], [], [], []
        for batch in batches:
            # skip batches with none of one label
            if (batch['label'] == 1).sum() == 0 or (batch['label'] == -1).sum() == 0:
                continue
            input_ids.append(batch['text']['input_ids'])
            attention_mask.append(batch['text']['attention_mask'])
            lengths.append(batch['text']['input_ids'].shape[0])
            new_batches.append(batch)
        batches = new_batches
        if len(batches) == 0:
            return None
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
        for length, batch in zip(lengths, batches):
            vecs1, vecs2 = allvecs1[offset:offset+length], allvecs2[offset:offset+length]
            offset += length
            predictions = torch.mm(vecs1, vecs2.transpose(0, 1))
            predictions = torch.sigmoid(predictions)
            pos_loss = - torch.log(predictions[batch['label'] == 1]).sum()
            neg_loss = - torch.log(1 - predictions[batch['label'] == -1]).sum()
            # loss = (pos_loss + neg_loss) / (batch['label'] != 0).sum() # unbalanced loss
            loss = ((pos_loss / (batch['label'] == 1).sum()) +
                    (neg_loss / (batch['label'] == -1).sum())) / 2 # balanced loss
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
    def __init__(self, from_pretrained='roberta-base', eq_means_lt=False):
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        self.eq_means_lt = eq_means_lt
    def __call__(self, instances):
        batch = dict(
            text=self.tokenizer([self.tokenizer.cls_token + ' ' + i['prompt'] for i in instances],
                                return_tensors='pt', padding=True, truncation=True),
        )
        if 'label' in instances[0].keys():
            labels = []
            for i, i1 in enumerate(instances):
                labels.append([])
                for j, i2 in enumerate(instances):
                    if self.eq_means_lt:
                        labels[-1].append(0 if i == j else 1 if i1['label'] > i2['label'] else -1)
                    else:
                        labels[-1].append(
                            0 if i == j or i1['label'] == i2['label'] else 1 if i1['label'] > i2['label'] else -1)
            batch['label'] = torch.tensor(labels, dtype=torch.float32)
        return batch
def setup_and_train_roberta_pair_ranking(get_full_df, task, checkpoint, key,
                                         epochs=10, batch_size=64, lr=1e-5, seed=0, setting=None,
                                         from_pretrained='roberta-base', no_training=False, eq_means_lt=False,
                                         no_labels=False):
    def get_df():
        df = get_full_df()
        return df[(df.task == task) & (df.checkpoint == checkpoint)]
#     dm = PromptsPerformanceData(get_df, key=key, batch_size=batch_size, collate_fn=PairRankingCollateFn(),
#                                 seed=seed,
#                                 setting=setting)
    dm = PromptsPerformanceDataByTestSubset(
        get_df, key=key, batch_size=batch_size, collate_fn=PairRankingCollateFn(
            from_pretrained=from_pretrained, eq_means_lt=eq_means_lt),
        seed=seed,
        setting=setting, no_labels=no_labels)
    dm.setup()
    trainer = pl.Trainer(
        gpus=1, max_epochs=epochs, log_every_n_steps=10, logger=WandbLogger(),
        check_val_every_n_epoch=8,
        callbacks=[
            # EarlyStopping(monitor="loss/val", mode="min", patience=100),
            ModelCheckpoint(dirpath=get_checkpoint_dir(), save_top_k=3, save_last=True,
                            monitor="acc/val", mode='max')
            # ModelCheckpoint(dirpath=get_checkpoint_dir(), save_top_k=3, save_last=True,
            #                 monitor="loss/val", mode='min')
        ],
    )
    model = StringPairRankingModel(lr=lr, from_pretrained=from_pretrained)
    freeze_roberta(model.model)
    if not no_training:
        trainer.fit(model, datamodule=dm,
            **(dict(ckpt_path=os.path.join(get_checkpoint_dir(), 'last.ckpt'))
               if wandb.run.resumed else {}))
        best_model_path = trainer.checkpoint_callback.best_model_path
    else:
        assert wandb.run.resumed
        last_checkpoint = torch.load(os.path.join(get_checkpoint_dir(), 'last.ckpt'))
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


def get_checkpoint_dir():
    return os.path.join('/'.join(wandb.run.dir.split('/')[:-3]), 'checkpoints', wandb.run.id)

def get_regcoef_dir():
    return os.path.join('/'.join(wandb.run.dir.split('/')[:-3]), 'regcoef', wandb.run.id)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--indir')
    parser.add_argument('--outdir')
    parser.add_argument('--mode')
    parser.add_argument('--task')
    parser.add_argument('--setting', default=None)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--metric', default='f1_normalized')
    parser.add_argument('--from_pretrained', default='roberta-base')
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--no_training', default=False, action='store_true')
    parser.add_argument('--eq_means_lt', default=False, action='store_true')
    parser.add_argument('--no_labels', default=False, action='store_true')
    cli_args = parser.parse_args()
    wandb.init(config=cli_args if cli_args.run_id is None else None, project="prompt_regression", dir=cli_args.outdir,
               **(dict(id=cli_args.run_id, resume="must") if cli_args.run_id is not None else {}))
    if cli_args.run_id is not None:
        # if continuing from a previous run, save old config and update using command line args
        old_configs = [] if 'old_configs' not in wandb.config.keys() else wandb.config['old_configs']
        old_configs.append(wandb.config)
        wandb.config.update(**cli_args)
        wandb.config['old_configs'] = old_configs
    args = Namespace(**wandb.config)
    get_full_df = GetFullDF(args.indir, tasks=(args.task,))
    print(get_breakdown(get_full_df()))
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(os.path.join(args.outdir, 'checkpoints')):
        os.mkdir(os.path.join(args.outdir, 'checkpoints'))
    if not os.path.exists(get_regcoef_dir()):
        os.makedirs(get_regcoef_dir())
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
        key = 'f1_normalized'
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
        key = args.metric
        trainer, dm, model = setup_and_train_roberta_regression(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key,
            epochs=300, no_training=args.no_training,
            batch_size=48, lr=args.lr,
            setting=args.setting, no_labels=args.no_labels)
        train_df, val_df = get_prediction_dfs_roberta_regression(trainer, dm, model)
        table = wandb.Table(columns=['split', 'predicted', 'real', 'len', 'corr', 'figure'])
        evaluate_prediction_df(train_df, key, 'acc_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'acc_normalized', 'val', table)
        evaluate_prediction_df(train_df, key, 'f1_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'f1_normalized', 'val', table)
        wandb.log({'table': table})
    elif args.mode == 'pairwise_comparator_ranking':
        key = args.metric
        trainer, dm, model = setup_and_train_roberta_pair_ranking(
            get_full_df, args.task, 'checkpoints/metaicl/hr_to_lr/model.pt', key,
            epochs=300, no_training=args.no_training,
            batch_size=48, lr=args.lr,
            setting=args.setting, eq_means_lt=args.eq_means_lt, no_labels=args.no_labels)
        val_df = get_roberta_pair_ranking(trainer, dm, model, split='val')
        train_df = get_roberta_pair_ranking(trainer, dm, model, split='train')
        table = wandb.Table(columns=['split', 'predicted', 'real', 'len', 'corr', 'figure'])
        evaluate_prediction_df(train_df, key, 'acc_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'acc_normalized', 'val', table)
        evaluate_prediction_df(train_df, key, 'f1_normalized', 'train', table)
        evaluate_prediction_df(val_df, key, 'f1_normalized', 'val', table)
        wandb.log({'table': table})
    else:
        raise NotImplementedError
    wandb.finish()

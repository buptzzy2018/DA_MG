#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import numpy as np
import pandas as pd
import random
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset


def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    audio, sr = load_wav_file(filename)
    audio = random_chunk(audio, max_frames, evalmode, num_eval)
    return audio, sr


def load_wav_file(filename):
    sr, audio = wavfile.read(filename)
    return audio, sr


def random_chunk(audio, max_frames, evalmode=False, num_eval=10):
    max_audio = max_frames
    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and num_eval == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return feat


def speed_perturb(waveform, sample_rate, label, num_spks):
    """ Apply speed perturb to the data.
    """
    speeds = [1.0, 0.9, 1.1]
    speed_idx = random.randint(0, 2)
    if speed_idx > 0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            torch.from_numpy(waveform[np.newaxis, :]), sample_rate,
            [['speed', str(speeds[speed_idx])], ['rate', str(sample_rate)]])
        waveform = wav.numpy()[0]
        label = label + num_spks * speed_idx

    return waveform, label


class AugmentWAV(object):
    def __init__(self, musan_data_list_path, rirs_data_list_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
        self.noiselist = {}

        df = pd.read_csv(musan_data_list_path)
        augment_files = df["utt_paths"].values
        augment_types = df["speaker_name"].values
        for idx, file in enumerate(augment_files):
            if not augment_types[idx] in self.noiselist:
                self.noiselist[augment_types[idx]] = []
            self.noiselist[augment_types[idx]].append(file)
        df = pd.read_csv(rirs_data_list_path)
        self.rirs_files = df["utt_paths"].values

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        audio = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True) + audio
        return audio.astype(np.int16).astype(float)

    def reverberate(self, audio):
        rirs_file = random.choice(self.rirs_files)
        fs, rirs = wavfile.read(rirs_file)
        rirs = np.expand_dims(rirs.astype(float), 0)
        rirs = rirs / np.sqrt(np.sum(rirs**2))
        if rirs.ndim == audio.ndim:
            audio = signal.convolve(audio, rirs, mode='full')[:,:self.max_audio]
        return audio.astype(np.int16).astype(float)



class Train_Dataset(Dataset):
    def __init__(self, data_list_path, nPerSpeaker,aug_prob, speed_perturb, max_frames, sample_rate, 
                 musan_list_path=None, rirs_list_path=None, eval_mode=False, data_key_level=0):
        # load data list
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        self.genre_list = df["genre"].values
        self.nPerSpeaker = nPerSpeaker

        self.speaker_number = len(np.unique(self.data_label))
        print("Train Dataset load {} speakers".format(self.speaker_number))
        print("Train Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = max_frames * sample_rate // 100

        if aug_prob > 0:
            self.augment_wav = AugmentWAV(musan_list_path, rirs_list_path, max_frames=self.max_frames)

        self.aug_prob = aug_prob
        self.speed_perturb = speed_perturb
        self.eval_mode = eval_mode
        self.data_key_level = data_key_level

        self.label_dict = {}
        for idx, speaker_label in enumerate(self.data_label):
            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []
            self.label_dict[speaker_label].append(idx)

        self.genre_label_dict = {}
        for idx, genre_label in enumerate(self.genre_list):
            if not (genre_label in self.genre_label_dict):
                self.genre_label_dict[genre_label] = []
            self.genre_label_dict[genre_label].append(idx)

        self.genre_number_dict = {}
        for key in self.label_dict.keys():
            index = len(set([self.genre_list[item] for item in self.label_dict[key]]))
            self.genre_number_dict[key] = index

    # sampler
    def __getitem__(self, indices):
        
        if type(indices) == int:
            index = indices
            audio, sr = load_wav_file(self.data_list[index])
            label = self.data_label[index]
            genre = self.genre_list[index]

            audio = random_chunk(audio, self.max_frames)
            if self.aug_prob > random.random():
                augtype = random.randint(1, 4)
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio)
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)

            if self.eval_mode:
                data_path_sp = self.data_list[index].split('/')
                data_key = data_path_sp[-1]
                for i in range(2, self.data_key_level + 1):
                    data_key = data_path_sp[-i] + '/' + data_key
                return torch.FloatTensor(audio), data_key

            return torch.FloatTensor(audio), label ,genre
        
        else:

            feat = []
            genre_feat = []
            for index in indices:
                audio, sr = loadWAV(self.data_list[index], self.max_frames)
                if self.aug_prob > random.random():
                    augtype = random.randint(1, 4)
                    if augtype == 1:
                        audio = self.augment_wav.reverberate(audio)
                    elif augtype == 2:
                        audio = self.augment_wav.additive_noise('music', audio)
                    elif augtype == 3:
                        audio = self.augment_wav.additive_noise('speech', audio)
                    elif augtype == 4:
                        audio = self.augment_wav.additive_noise('noise', audio)
                feat.append(audio)
                genre = self.genre_list[index]
                genre_feat.append(genre)

            feat = np.concatenate(feat, axis=0)
            return torch.FloatTensor(feat), self.data_label[index],torch.tensor(genre_feat)


    def __len__(self):
        return int(len(self.data_list)/self.nPerSpeaker)


class Dev_Dataset(Dataset):
    def __init__(self, data_list_path, eval_frames, num_eval=0, **kwargs):
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Dev Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Dev Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio, sr = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return int(len(self.data_list)/16)


class Test_Dataset(Dataset):
    def __init__(self, data_list, eval_frames, num_eval=0, **kwargs):
        # load data list
        self.data_list = data_list
        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio, sr = loadWAV(self.data_list[index][1], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_list[index][0]

    def __len__(self):
        return len(self.data_list)


def round_down(num, divisor):
    return num - (num % divisor)



class Train_Sampler_genre(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.genre_label_dict = data_source.genre_label_dict
        
    def __iter__(self):

        print('Train_Sampler_genre.........')
        dict_copy = self.genre_label_dict.copy()
        #字典的key为场景，value 为一个二维数组，第一个维度为这个场景的说话人数量，第二个维度是这个场景，这个说话人的 utts的索引 
        for key,value1 in dict_copy.items():
            dict1 = {}
            for i in value1:
                if not (self.data_source.data_label[i] in dict1):
                    dict1[self.data_source.data_label[i]] = []
                dict1[self.data_source.data_label[i]].append(i)
            
            filtered_dict = {key1: value for key1, value in dict1.items() if len(value) >= self.nPerSpeaker}
            if len(filtered_dict.keys())<self.batch_size/2:
                del self.genre_label_dict[key]
            else:
                self.genre_label_dict[key] = filtered_dict   

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        # 每个batch中,每个场景采样的说话人数
        Batch = int(self.batch_size/2)
        
        dictkeys = list(self.genre_label_dict.keys())
        Genre_num = len(dictkeys)

        while Genre_num>=2:
            # 每个场景的说话人数量设置为 nPerSpeaker
            # 1、每次随机选取两个场景作为 sorce domain 和 target domain
            random_genre = random.sample(dictkeys, 2)
            flag =True
            first = []
            
            list_first = list(self.genre_label_dict[random_genre[0]].keys())
            speakers_first = random.sample(list_first ,Batch)
            for spk in speakers_first:
                first_index = random.sample(self.genre_label_dict[random_genre[0]][spk], self.nPerSpeaker) 
                first.append(first_index)
                self.genre_label_dict[random_genre[0]][spk] = list(set(self.genre_label_dict[random_genre[0]][spk]) - set(first_index))
                if len(self.genre_label_dict[random_genre[0]][spk]) < self.nPerSpeaker:
                    self.genre_label_dict[random_genre[0]].pop(spk)

            list_second = list(set(self.genre_label_dict[random_genre[1]].keys()) - set(speakers_first))

            List = dictkeys.copy()
            List.remove(random_genre[0])
            while len(list_second) < Batch:
                List.remove(random_genre[1])
                if len(List) == 0:
                    flag = False
                    break
                random_genre[1] = random.choice(List)
                list_second = list(set(self.genre_label_dict[random_genre[1]].keys()) - set(speakers_first))

            if flag:
                second = []
                speakers_second = random.sample(list_second ,Batch)
                for spk in speakers_second:
                    second_index = random.sample(self.genre_label_dict[random_genre[1]][spk], self.nPerSpeaker) 
                    second.append(second_index)
                    self.genre_label_dict[random_genre[1]][spk] = list(set(self.genre_label_dict[random_genre[1]][spk]) - set(second_index))
                    if len(self.genre_label_dict[random_genre[1]][spk]) < self.nPerSpeaker:
                        self.genre_label_dict[random_genre[1]].pop(spk)

                finial = [item for pair in zip(first, second) for item in pair]
                flattened_list.append(finial)
 
                if (sum(len(value) for value in self.genre_label_dict[random_genre[0]].values()) < Batch*self.nPerSpeaker) or (len(self.genre_label_dict[random_genre[0]].keys()) < Batch):
                    Genre_num = Genre_num - 1
                    dictkeys.remove(random_genre[0])
                
                if (sum(len(value) for value in self.genre_label_dict[random_genre[1]].values()) < Batch*self.nPerSpeaker) or (len(self.genre_label_dict[random_genre[1]].keys()) < Batch):
                    Genre_num = Genre_num - 1
                    dictkeys.remove(random_genre[1])

            else:
                if (sum(len(value) for value in self.genre_label_dict[random_genre[0]].values()) < Batch*self.nPerSpeaker) or (len(self.genre_label_dict[random_genre[0]].keys()) < Batch):
                    Genre_num = Genre_num - 1
                    dictkeys.remove(random_genre[0])

        random.shuffle(flattened_list)
        flattened_list = np.array(flattened_list).reshape(-1 , self.nPerSpeaker).tolist()
        print('np.array(flattened_list.shape)', np.array(flattened_list).shape)
        return iter(flattened_list)


    def __len__(self):
        return len(self.data_source)


class Train_Sampler_speaker(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.genre_number_dict = data_source.genre_number_dict
        self.genre_list = data_source.genre_list


    def __iter__(self):

        print('Train_Sampler_speaker.........')
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data = self.label_dict[key]
            numSeg = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            if self.genre_number_dict[key] > 1:
                dict1 = {}
                for idx,index in enumerate(data):
                    if not (self.genre_list[index] in dict1):
                        dict1[self.genre_list[index]] = []
                    dict1[self.genre_list[index]].append(idx)
                # dict1 里面存的 key 值为场景 ，value值为对应语音的索引
                # 2、将 n个子列表进行打乱
                for key1,value in dict1.items():
                    random.shuffle(value)
                rp = []
                genre_num = self.genre_number_dict[key]
                while genre_num > 1:
                    key_list = list(dict1.keys())
                    choices = random.sample(list(range(genre_num)), 2)
                    first_index = random.choice(dict1[key_list[choices[0]]])
                    second_index = random.choice(dict1[key_list[choices[1]]])
                    tuple_index = np.array([first_index,second_index])
                    rp.append(tuple_index)
                    dict1[key_list[choices[0]]].remove(first_index)
                    dict1[key_list[choices[1]]].remove(second_index)

                    if len(dict1[key_list[choices[0]]]) == 0:
                        del dict1[key_list[choices[0]]]
                        genre_num = genre_num - 1

                    if len(dict1[key_list[choices[1]]]) == 0:
                        del dict1[key_list[choices[1]]]
                        genre_num = genre_num - 1
                if genre_num != 0:
                    # print('genre_num', genre_num)
                    # print('剩下一个场景')
                    # print(dict1.keys())
                    # print(list(dict1.keys()))
                    last = dict1[list(dict1.keys())[0]]
                    numSeg2 = round_down(len(last), self.nPerSpeaker)
                    tuple_list = lol(np.random.permutation(len(last))[:numSeg2], self.nPerSpeaker)
                    rp.extend(tuple_list)  
            else:
                # print('场景数为1')
                rp = lol(np.random.permutation(len(data))[:numSeg],self.nPerSpeaker)

            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])
        ## Data in random order
        mixid = np.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []
        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        print('np.array([flattened_list[i] for i in mixmap]).shape',np.array([flattened_list[i] for i in mixmap]).shape)

        return iter([flattened_list[i] for i in mixmap])

    def __len__(self):
        return len(self.data_source)

if __name__ == "__main__":
    data, sr = loadWAV("test.wav", 100, evalmode=True)
    print(data.shape)
    data, sr = loadWAV("test.wav", 100, evalmode=False)
    print(data.shape)

    def plt_wav(data, name):
        import matplotlib.pyplot as plt
        x = [ i for i in range(len(data[0])) ]
        plt.plot(x, data[0])
        plt.savefig(name)
        plt.close()

    plt_wav(data, "raw.png")
    
    aug_tool = AugmentWAV("data/musan_list.csv", "data/rirs_list.csv", 100)

    audio = aug_tool.reverberate(data)
    plt_wav(audio, "reverb.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "music.png")

    audio = aug_tool.additive_noise('speech', data)
    plt_wav(audio, "speech.png")

    audio = aug_tool.additive_noise('noise', data)
    plt_wav(audio, "noise.png")

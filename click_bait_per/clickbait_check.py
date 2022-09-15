from kobert_transformers import get_tokenizer
from model.utils import pytorch_cos_sim

from tqdm.notebook import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


import pandas as pd
import numpy as np

from transformers import ElectraModel, ElectraConfig, ElectraTokenizer, AdamW, get_cosine_schedule_with_warmup
from data.dataloader import convert_to_tensor, example_model_setting
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, num_cls):
        super(Model, self).__init__()
        self.dim = 768*4
        self.hidden = 64
        self.out = nn.Sequential(nn.Linear(self.dim, self.hidden),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, embedding):
        return self.out(embedding)

class ContrastiveDataset(Dataset):
    def __init__(self, encoder, transform, device, sentence1, sentence2, label, max_len=512):
        self.tokenizer = get_tokenizer()
        self.s1 = []
        self.s2 = []
        self.label = []
        self.max_len = max_len
        assert len(sentence1) == len(sentence2) 
            
        s1 = convert_to_tensor(sentence1, transform)
        s2 = convert_to_tensor(sentence2, transform)
        self.label = label
            
        s1_input = {}
        s2_input = {}
        
        s1_source = []
        s1_seg = []
        s1_val = []
        
        s2_source = []
        s2_seg = []
        s2_val = []
        
        for idx in tqdm(range(len(sentence1))):
            s1_source.append(s1['source'][idx])
            s1_seg.append(s1['segment_ids'][idx])
            s1_val.append(s1['valid_length'][idx])
            
            s2_source.append(s2['source'][idx])
            s2_seg.append(s2['segment_ids'][idx])
            s2_val.append(s2['valid_length'][idx]) 
        
        s1_input['source'] = s1_source
        s1_input['segment_ids'] = s1_seg
        s1_input['valid_length'] = s1_val
        
        s2_input['source'] = s2_source
        s2_input['segment_ids'] = s2_seg
        s2_input['valid_length'] = s2_val
                           
        self.s1 = s1_input
        self.s2 = s2_input
                        
    def __len__(self):
        return len(self.s1['source'])
    
    def __getitem__(self, idx):
        return self.s1['source'][idx], self.s1['segment_ids'][idx], self.s1['valid_length'][idx], self.s2['source'][idx], self.s2['segment_ids'][idx], self.s2['valid_length'][idx], torch.tensor(self.label[idx], dtype=torch.long)

def create_data_loader(encoder, transform, device, df, max_len, batch_size, num_workers):
    cd = ContrastiveDataset(
        encoder = encoder,
        transform = transform,
        device = device,
        sentence1=df.iloc[:,1].to_numpy(),
        sentence2=df.iloc[:,2].to_numpy(),
        label = df.iloc[:,3].to_numpy(),
        max_len=50
    )
  
    return DataLoader(
        cd,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True
    )

def make_tensorloader(encoder, device, data_loader, batch_size, train=False):
    # train=False로 했지만 train은 True하고 test일때 False
    output = []
    labels = []
    title = {}
    body = {}
    
    encoder.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            s1_source = data[0]
            s1_segment = data[1]
            s1_valid = data[2]
            s2_source = data[3]
            s2_segment = data[4]
            s2_valid = data[5]
            label = data[6]
            
            title['source'] = s1_source
            title['segment_ids'] = s1_segment
            title['valid_length'] = s1_valid
            body['source'] = s2_source
            body['segment_ids'] = s2_segment
            body['valid_length'] = s2_valid
                      
            s1 = encoder.encode(title, device)
            s2 = encoder.encode(body, device)      
            
            s = torch.cat([s1, s2, abs(s1-s2), s1*s2], dim=1)
            
            output.append(s)            
            labels.append(label)
            
            del s, label, s1, s2, s1_source, s1_segment, s1_valid, s2_source, s2_segment, s2_valid
            
        output = torch.cat(output, dim=0).contiguous().squeeze()#.cuda()
        labels = torch.cat(labels)
        
    print('output:', output.shape)
    print('labels:', labels.shape)
    linear_ds = TensorDataset(output, labels)
    linear_loader = DataLoader(linear_ds, batch_size=batch_size, shuffle=train, drop_last=True)
    
    return linear_loader


MAX_LEN = 512
batch_size = 16
num_workers=16
device = torch.device('cpu')

class_names = ['NonClickBait', 'ClickBait']
PATH = './trained_model_simcse_skt/downstream/final_model.bin'

encoder_ckpt = './output/nli_checkpoint.pt'
encoder, transform, device = example_model_setting(encoder_ckpt)

classifier = Model(len(class_names))

if isinstance(classifier, nn.DataParallel):  # GPU 병렬사용 적용
    encoder.load_state_dict(PATH)
else: # GPU 병렬사용을 안할 경우
    state_dict = torch.load(PATH) 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`  ## module 키 제거
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)'
    
classifier.to(device)

df = pd.read_csv('./data/test_dataset.csv')
content = "슈퍼주니어 신동이 팬들의 충격적인 선물을 공개했다. 지난 7일 방송된 MBC 예능 ‘라디오스타’(이하 ‘라스’)에서는 추석특집 ‘흥! 끌어올려’ 특집으로 진성, 김호중, 금잔디, 신동, ITZY(있지) 채령이 등장했다. 이날 김호중은 “앨범도 나오고 광고도 찍고 오늘 소집해제 후 첫 토크쇼”라고 말했다. 진성은 김호중에게 용돈을 준 적이 있다며 “내가 그때 돈이 좀 있었다. 그걸 전체 다 빼주면 나도 또 그러니까 일부만 뺐는데 4장 정도가 됐다”며 “5만 원짜리 4장이다. 20만 원”이라고 밝혔다. 이에 김호중은 이날 진성이 입은 의상을 맞춰줬다고. 진성은 “이게 기성복이 아니고 맞춤이다. 바지, 구두까지”라며 자랑했다. 신동은 소속사와의 재계약에 대해 언급했다. 슈퍼주니어-T에서 ‘똑똑똑’ 등 트로트 앨범을 냈던 경험이 있는 신동은 “트로트 가수분들의 뮤직 비디오 제안이 많이 온다”고 밝히며 “지금 재계약 시즌이다. 이번에도 재계약이 잘 돼서 더 오래 하고 싶다. 재계약금을 조금 더 불렀는데 왔다 갔다 하고 있다. 방송에서 이야기하면 더 많이 주시지 않을까 싶다”라고 말했다. SM 이사 자리가 탐나지 않냐고 묻자 신동은 “강타, 보아, 김민종 이사님이 계시는데 이사 욕심은 없다. 후배 양성 욕심이 있다. 그래서 이수만 선생님께 연락드려서 경영에 대해 배울 수 있냐고 물었다. 그랬더니 같이 밥 먹자고 하셨다”라고 밝혔다. 개인적으로 회사를 차릴 생각이 있냐는 질문에는 “난 SM 안에서 하고 싶다. 굳이 제 돈으로 하냐. 회사 자본이 있는데”라고 강조했다. 신동은 팬들의 격한 사랑에 당황한 적이 있다고 털어놓기도. 그는 “남미에서 공연 도중 한 팬분이 이렇게 (상의 사이로) 손을 넣더니 속옷을 풀어 던지더라”고 말했다. 이는 남미의 독특한 응원문화라고. 신동은 “당시는 처음이라 몰랐다. 공연을 하다보니 그게 있어서 ‘이걸 왜?’하면서 놀라는 반응을 했는데 그 반응에 신이 났는지 다음 공연에는 더 많이 챙겨와서 이것저것을 던지더라”고 회상했다. 이어 그는 “가방에 이만큼 챙겨와 각종 속옷을 던지더니 그 다음부터는 더 센 것들도 던졌다. 피임 기구까지 던지더라”며 “그래서 한번은 ‘이제 그만 던지라’고 얘기했다. 저희가 그걸 챙기는 것도 이상하지 않냐”고 해 웃음을 자아냈다. 신동도 팬들에게 의도치 않게 뭔가를 던진 적이 있다고. 그는 “내 앞니가 래미네이트다. 본뜨기 전 임시 치아를 일주일간 했는데 공연 중 마이크로 앞니를 툭 쳐서 앞니 4개가 날아갔다”며 “놀라서 주워 끼웠는데 다행히 끼워졌다”고 말했다."
df.loc[15,'summary'] = content # content에 기사 본문 
title = "[종합]`속옷 풀어 던져, 피임기구까지 투척`…신동, SM 재계약 불발? `조금 더 불렀는데` ('라스')"
df.loc[15,'title'] = title # title에 기사 제목
df.loc[15,'clickbait'] = 0.0

verify_data_loader = create_data_loader(encoder, transform, device, df, MAX_LEN, batch_size, num_workers)
verifyloader = make_tensorloader(encoder, device, verify_data_loader, batch_size=batch_size)

for embedding,label in tqdm(verifyloader):
    with torch.no_grad():
        out = classifier(embedding)
        preds = torch.argmax(out,dim=1)
        
        out = out.cpu().detach()
        print(F.softmax(out[15], dim=0).max().item()) # 출력값 => 확률값 
def split_wav(data, sample_rate, start, end):
  start *= sample_rate
  end *= sample_rate
  return data[start:end]

pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

DATA_DIR_1 = "/media/sungshin/새 볼륨/"
DATA_DIR_2="/media/sungshin/새 볼륨/"

# Data set list, include (raw data, mfcc data, y data)
trainset = []
testset = []

# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
train_X = []
train_mfccs = []

test_X = []
test_mfccs = []

frame_length = 0.025
frame_stride = 0.0010

# train data를 넣는다.
for filename in os.listdir(DATA_DIR_1 + "trainset2/"):
    filename = normalize('NFC', filename)
    try:
        # wav 포맷 데이터만 사용
        if '.wav' not in filename in filename:
            continue

        wav, sr = librosa.load(DATA_DIR_1+ "trainset2/" + filename, sr=16000)
        wav=split_wav(wav, sr, 450, 462) #12sec
        mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40, n_fft=400, hop_length=160) #특징 40개 추출
        padded_mfcc = pad2d(mfcc, 1200) #12초

        # 추임새 별로 dataset에 추가
        if filename[1] == 'C': #충청도
            trainset.append((padded_mfcc, 0))
        elif filename[1] == 'J': #전라도
            trainset.append((padded_mfcc, 1))
        elif filename[1] == 'K': #경상도
            trainset.append((padded_mfcc, 2))
        elif filename[1] == 'G':  #강원도
            trainset.append((padded_mfcc, 3))
        elif filename[1] == 'Z':  # 제주도
            trainset.append((padded_mfcc, 4))
    except Exception as e:
        print(filename, e)
        raise

# test data를 넣는다.
for filename in os.listdir(DATA_DIR_2 + "testset2/"):
    filename = normalize('NFC', filename)
    try:
        # wav 포맷 데이터만 사용
        if '.wav' not in filename in filename:
            continue

        wav, sr = librosa.load(DATA_DIR_2 + "testset2/" + filename, sr=16000)
        wav=split_wav(wav, sr, 450, 462) #12sec
        mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40, n_fft=400, hop_length=160)
        padded_mfcc = pad2d(mfcc, 1200)
        # print(filename)
        # 추임새 별로 test dataset에 추가
        if filename[1] == 'C':
            testset.append((padded_mfcc, 0))
        elif filename[1] == 'J':
            testset.append((padded_mfcc, 1))
        elif filename[1] == 'K':
            testset.append((padded_mfcc, 2))
        elif filename[1] == 'G':
            testset.append((padded_mfcc, 3))
        elif filename[1] == 'Z': 
            testset.append((padded_mfcc, 4))
    except Exception as e:
        print(filename, e)
        raise

#파일 생성
train_x = [a for (a,b) in trainset]
train_y = [b for (a,b) in trainset]
train_x = np.array(train_x)
train_y = np.array(train_y)
np.save('train_x_test.npy',train_x)
np.save('train_y_test.npy',train_y)

test_x = [a for (a,b) in testset]
test_y = [b for (a,b) in testset]
test_x = np.array(test_x)
test_y=np.array(test_y)
np.save('test_y_test.npy',test_y)
np.save('test_x_test.npy',test_x)

train_y_cat = to_categorical(np.array(train_y)) #(0,1,1,0) -> ((1,0), (0,1), (0,1), (1,0))
test_y_cat = to_categorical(np.array(test_y))
np.save('train_y_cat_test.npy',train_y_cat)
np.save('test_y_cat_test.npy',test_y_cat)

import email
import os
import csv
import html2text
from sklearn.model_selection import train_test_split
import pandas as pd


SPAM_DIRS = {
  'spamassassin': {
    'spam': ['spam', 'spam_2'],
    'ham': ['easy_ham', 'hard_ham', 'easy_ham_2'],
  },
  'enron': {
    'spam': ['BG', 'GP', 'SH'],
    'ham': ['beck-s', 'farmer-d', 'kaminski-v', 'kitchen-l', 'lokay-m', 'williams-w3'],
  },
}

OUT_CSV = 'dataset.email.%d.csv'


def save_dir(msg_type, dir):
  for sub in os.listdir(dir):
    if os.path.isdir(dir + '/' + sub):
      save_dir(msg_type, dir + '/' + sub)
    else:
      with open(dir + '/' + sub, 'rb') as f:
        raw = f.read()
        msg = email.message_from_bytes(raw)
        save_doc(msg_type, msg, out)


def save_doc(msg_type, msg, out):
  global i
  txt = msg.get_payload(decode=True)

  if txt is None:
    for submsg in msg.get_payload():
      save_doc(msg_type, submsg, out)
  else:
    if b'PGP' not in txt:
      csv.writer(out, escapechar='\\').writerow([msg_type, (str(msg['Subject'])+' ' if msg['Subject'] else '') + html2text.html2text(txt.decode('utf-8', 'ignore')).replace('\n', ' ').strip()])


d = 1
for dataset in SPAM_DIRS.keys():
  with open(OUT_CSV % d, 'w') as out:
    csv.writer(out).writerow(['label', 'text'])
    for msg_type in SPAM_DIRS[dataset].keys():
      for dir in SPAM_DIRS[dataset][msg_type]:
        save_dir(msg_type, dataset + '/' + dir)
  d += 1


# df = pd.read_csv(f"dataset.sms.1.csv")
# df_train, df_test = train_test_split(df, test_size=0.3)
# df_train.to_csv(f"dataset.sms.train.csv", index=False)
# df_test.to_csv(f"dataset.sms.test.csv", index=False)

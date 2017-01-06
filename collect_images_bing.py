#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 実行方法
# コマンドラインから下記のようにして実行する．
# ただし，引数dirはget_IMGを用いない(画像を保存しない)場合は指定しない．
# collect_img.py
# $ python collect_img.py query dir
# query: 欲しい画像の検索ワード (e.g. leopard)
# dir: 画像保存先ディレクトリ (./img/*)
# wgetを使っているため、使えるのはLinuxのみ

import sys
import os
import re
import commands as cmd
import pdb

# read config.ini
import ConfigParser
inifile = ConfigParser.SafeConfigParser()
inifile.read('./config.ini')
NUM_CLASSES = int(inifile.get("settings", "num_classes"))
DOWNLOAD_LIMIT = int(inifile.get("settings", "download_limit"))

# クエリ検索したHTMLの取得
def get_HTML(query):

    # html = cmd.getstatusoutput("wget -O - https://www.bing.com/images/search?q=" + query + "&qft=+filterui:imagesize-large&FORM=R5IR3")

    html = cmd.getstatusoutput("wget -O - https://www.bing.com/images/search?q=" + query + "&qft=+filterui:age-lt43200+filterui:photo-photo&FORM=R5IR22")

    # html = cmd.getstatusoutput("wget -O - https://www.bing.com/images/search?q=" + query)

    return html

# jpg画像のURLを抽出
def extract_URL(html):

    url = []
    sentences = html[1].split('\n')
    # with open("bing_html.txt", "w") as f:
    #     for line in sentences:
    #         f.write(line+"\n")

    ptn = re.compile('<a class="thumb" target="_blank" href="(.*?).jpg')
    count = 0
    for sent in sentences:
        sents = sent.split('div class=')
        for s in sents:
            search = re.search(ptn, s)
            if search is None:
                print 'Not match'
            else:
                if count==DOWNLOAD_LIMIT:
                    break
                print search.group(1)
                url.append(search.group(1)+".jpg")
                count += 1
        if count==DOWNLOAD_LIMIT:
            print "reach the DOWNLOAD_LIMIT"
            break

    return url

# ローカルに画像を保存
def get_IMG(dir,url):

    for u in url:
        try:
            os.system("wget -P  " + dir + " " + "–no-clobber -N "+ u)
        except:
            continue


if __name__ == "__main__":

    argvs = sys.argv # argvs[1]: 画像検索のクエリ, argvs[2]: 保存先のディレクトリ(保存したい時のみ)
    query = argvs[1] # some images  e.g. leopard

    html = get_HTML(query)

    url = extract_URL(html)

    for u in url:
        print u

    # 画像をローカルに保存したいときに有効にする
    get_IMG(argvs[2],url)

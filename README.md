#Assignment I
<li>題目：鴨子辨識</li>
<li>環境：pycharm</li>
<li>使用套件</li><p>

>numpy<p>
>sklearn<p>
>cv2<p>
>matplotlib<p>
>scipy<p>

<br><li>演算法：
將鴨子的資料切下一小塊並轉為灰階，
首先預設像素值小於225的像素為真值(鴨子為白色)；
其他像素質則為假值，代表非鴨子。
將兩類資料非配給貝氏分類器的模型進行訓練，
再將欲預測的資料(full_duck)丟進去進行預測。</li>

<br><li>程式架構：
首先切割出要訓練的資料集，
分別是鴨子的資料集和非鴨子的資料集。
<br>
![Alt text](https://i.imgur.com/9mNX6Dx.jpg)<br>
![Alt text](https://i.imgur.com/KBhvHCq.jpg)<br>
<br>
開檔完後將資料轉為灰階方便後續處理。<br>
![Alt text](https://i.imgur.com/hyRStMB.jpg)<br>
<br>
這邊使用的方式是先將鴨子的訓練資料轉為numpy的陣列，
並設像素值225為門檻。
低於225的像素通通設為非鴨子的類別，
否則視為鴨子。<br>
![Alt text](https://i.imgur.com/aQmsDYC.jpg)<br>
非鴨子的圖片資料以相同的方式進行處理，
差別在於非鴨子的圖片中因為沒有存在鴨子，
所以通通設為0。
<br><br>
再叫出一次原圖，
一樣轉為灰階後再攤平成一維，
以便進行訓練。<br>
![Alt text](https://i.imgur.com/5cTEW2I.jpg)<br>
<br>首先將資料分割進行測試，
正解為像素質只有0和1的array，
切割0.1為測試集。
再建立一個naive bayes的模型，
先以切割的鴨子資料進行訓練，
然後使用partial_fit以沒有鴨子的資料集嘗試優化模組(後面會解釋原因)。<br>
![Alt text](https://i.imgur.com/asTtb7q.jpg)<br>
<br>進行評分後可以看見模組隊訓練資料集裡切割出來的測試資料預測分數為：<br>
![Alt text](https://i.imgur.com/SMWTAuU.jpg)<br>
<br>可以將兩個label的平均數和標準差取出並建立高斯曲線觀察兩個類別的分布。
![Alt text](https://i.imgur.com/AamVkHg.jpg)<br>
![Alt text](https://i.imgur.com/XTsNYoW.jpg)<br>
<br>模組訓練好後，
就可以把full_duck的圖片丟進來進行預測。
像上述步驟一樣進行圖像的處理，
然後將攤平的full_duck陣列進行預測。
因為預測出來的值為0或1，
所以使用迴圈將label為1(鴨子)的像素設為255並存成圖片。<br>
![Alt text](https://i.imgur.com/zEdVAG5.jpg)<br>
<br>預測出來的圖片為：<br>
![Alt text](https://i.imgur.com/9xaoA4F.jpg)<br></li>
<br><li>誤差：鴨子是利用像素進行預測，
所以顏色較近的鵝卵石會被模型誤判。
為了克服這個問題，
在建立模型時將沒有鴨子的資料一起訓練了，
解果比原本好了一些但仍無法完全去除所有誤判的鵝卵石。</li>

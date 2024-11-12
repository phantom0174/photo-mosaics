## 圖片馬賽克

關鍵字: photomosaics machine learning

這啥:
用不同場景的圖片去拼湊/擬合出指定圖片
查了一下這咚咚
發現原理會用到接下來要教的KNN (K-nearest neighbors
不用用到很後面的東西
所以感覺可做

之前看動漫官方粉專都會傳這種東西owo

訓練資料?:
如果要合成日常照片
就直接抓我們自己的相簿當成training data

如果像是要合成動漫角色圖
就直接抓原影片再把轉場後的特徵圖片全抓成dataframe
當成訓練資料
再加個educational purpose only就好

refs:
- https://zh.wikipedia.org/zh-tw/相片馬賽克
- https://github.com/worldveil/photomosaic
- https://www.lambertleong.com/projects/photo-mosaic
- https://applealmond.com/posts/215803

靈感來源:
- (genetic algo) https://www.youtube.com/watch?v=6aXx6RA1IK4
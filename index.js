const path = require('path');
const cv = require('opencv4nodejs');

const src = cv.imread(path.resolve(__dirname, './data/hongkong.bmp'));
const rows = src.rows;
const cols = src.cols;
const w = 0.95;
const t0 = 0.1;
const minMat = new cv.Mat(rows, cols, cv.CV_8UC1);
const trans32F = new cv.Mat(rows, cols, cv.CV_32FC1);
const dst = new cv.Mat(rows, cols, cv.CV_8UC3);

const startTime = new Date().getTime()
// 获取最小值图
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    const [B, G, R] = src.atRaw(r, c);
    const minV = Math.min(B, G, R);
    minMat.set(r, c, minV);
  }
}

// 最小值滤波获取暗通道图
const k = 7;
const block = 2 * 7 + 1;
let g = [], h = [], hFlip = [], temp = [];
for (let r = 0; r < rows; r++) {
  for (let c = cols - 1; c >= 0; c--) {
    temp[cols - 1 - c] = minMat.atRaw(r, c);
  }

  for (let c = 0; c < cols; c++) {
    if (c % block == 0) {
      g[c] = minMat.atRaw(r, c);
      hFlip[c] = temp[c];
    } else {
      g[c] = Math.min(g[c - 1], minMat.atRaw(r, c));
      hFlip[c] = Math.min(hFlip[c - 1], temp[c]);
    }
  }
  for (let c = 0; c < cols; c++) {
    h[c] = hFlip[cols - 1 - c];
  }

  for (let c = 0; c < cols; c++) {
    if (c < k) {
      minMat.set(r, c, Math.min(g[c + k]));
    } else if (c >= cols - k) {
      minMat.set(r, c, Math.min(h[c - k]));
    } else {
      minMat.set(r, c, Math.min(g[c + k], h[c - k]));
    }
  }
}

g = []; h = []; hFlip = []; temp = [];
for (let c = 0; c < cols; c++) {
  for (let r = rows - 1; r >= 0; r--) {
    temp[rows - 1 - r] = minMat.atRaw(r, c);
  }

  for (let r = 0; r < rows; r++) {
    if (r % block == 0) {
      g[r] = minMat.atRaw(r, c);
      hFlip[r] = temp[r];
    } else {
      g[r] = Math.min(g[r - 1], minMat.atRaw(r, c));
      hFlip[r] = Math.min(hFlip[r - 1], temp[r]);
    }
  }
  for (let r = 0; r < rows; r++) {
    h[r] = hFlip[rows - 1 - r];
  }

  for (let r = 0; r < rows; r++) {
    if (r < k) {
      minMat.set(r, c, Math.min(g[r + k]));
    } else if (c >= cols - k) {
      minMat.set(r, c, Math.min(h[r - k]));
    } else {
      minMat.set(r, c, Math.min(g[r + k], h[r - k]));
    }
  }
}

const darkChannel8U = minMat.copy();
// for (let r = 0; r < rows; r++) {
//   for (let c = 0; c < cols; c++) {
//     let minV = 255;
//     for (let i = r - k; i < r + k; i++) {
//       for (let j = c - k; j < c + k; j++) {
//         if (i < 0 || j < 0 || i >= rows || j >= cols) {
//           continue;
//         }
//         if (minV > minMat.atRaw(i, j)) {
//           minV = minMat.atRaw(i, j);
//         }
//       }
//     }
//     darkChannel8U.set(r, c, minV);
//   }
// }

// 根据暗通道图获取大气光值A
let sortArray = [];
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    sortArray.push({
      r: r,
      c: c,
      val: darkChannel8U.atRaw(r, c)
    })
  }
}

sortArray.sort(function(a, b) {
  return a.val < b.val;
})

let sumB = 0, sumG = 0, sumR = 0;
for (let i = 0; i < (rows * cols / 1000); i++) {
  const [B, G, R] = src.atRaw(sortArray[i].r, sortArray[i].c);
  sumB += B;
  sumG += G;
  sumR += R;
}

const Ab = sumB / (rows * cols / 1000);
const Ag = sumG / (rows * cols / 1000);
const Ar = sumR / (rows * cols / 1000);
const A = parseInt(0.1140 * Ab + 0.5870 * Ag + 0.2989 * Ar);
// const A = 220;
console.log('A =', A);

// 获取透射率图
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    let t = 1.0 - (w * darkChannel8U.atRaw(r, c) / A);
    t < 0 && (t = 0);
    trans32F.set(r, c, t);
  }
}

// 对透射率图进行引导滤波
const src32F = src.convertTo(cv.CV_32F, 1.0/255, 0.0);
const guideTrans32F = trans32F.guidedFilter(src32F, 100, 0.01, -1);

// Gamma矫正
const gama = 0.8;
const GammaTable = [];
for (let i = 0; i < 256; i++) {
  let fT = (i + 0.5)/255;
  fT = Math.pow(fT, gama);
  let iT = parseInt(fT * 255);
  iT > 255 && (iT = 255);
  iT < 0 && (iT = 0); 
  GammaTable.push(iT);
}

// 获取去雾后图像
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    const [B, G, R] = src.atRaw(r, c);
    const t = Math.max(guideTrans32F.atRaw(r, c), t0);
    let dstB = parseInt((B - Ab) / t + Ab);
    let dstG = parseInt((G - Ag) / t + Ag);
    let dstR = parseInt((R - Ar) / t + Ar);
    dstB > 255 && (dstB = 255);
    dstG > 255 && (dstG = 255);
    dstR > 255 && (dstR = 255);
    dstB < 0 && (dstB = 0);
    dstG < 0 && (dstG = 0);
    dstR < 0 && (dstR = 0);
    // dst.set(r, c, [dstB, dstG, dstR]);
    dst.set(r, c, [GammaTable[dstB], GammaTable[dstG], GammaTable[dstR]]);
  }
}

const endTime = new Date().getTime()
console.log(`运行时间为: ${(endTime - startTime) / 1000}s`)
cv.imshowWait("src", src);
// cv.imshowWait("darkChannel", minMat);
// cv.imshowWait("trans", trans32F);
// cv.imshowWait("guided", guideTrans32F);
cv.imshowWait("dst", dst);
cv.imwrite('./data/src.jpg', src);
cv.imwrite('./data/dst.jpg', dst);
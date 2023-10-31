p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def AveragePooling2DInt (nRows, nCols, nChannels, poolSize, strides, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [[[0 for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    remainder = [[[None for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    for i in range((nRows-poolSize)//strides + 1):
        for j in range((nCols-poolSize)//strides + 1):
            for k in range(nChannels):
                for x in range(poolSize):
                    for y in range(poolSize):
                        out[i][j][k] += input[i*strides+x][j*strides+y][k]
                remainder[i][j][k] = str(out[i][j][k] % poolSize**2 % p)
                out[i][j][k] = str(out[i][j][k] // poolSize**2 % p)
    return Input, out, remainder

def BatchNormalizationInt(nRows, nCols, nChannels, n, X_in, a_in, b_in):
    X = [[[str(X_in[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    A = [str(a_in[k] % p) for k in range(nChannels)]
    B = [str(b_in[k] % p) for k in range(nChannels)]
    out = [[[None for _ in range(nChannels)] for _ in range(nCols)] for _ in range(nRows)]
    remainder = [[[None for _ in range(nChannels)] for _ in range(nCols)] for _ in range(nRows)]
    for i in range(nRows):
        for j in range(nCols):
            for k in range(nChannels):
                out[i][j][k] = (X_in[i][j][k]*a_in[k] + b_in[k])
                remainder[i][j][k] = str(out[i][j][k] % n)
                out[i][j][k] = str(out[i][j][k] // n % p)
    return X, A, B, out, remainder

def Conv1DInt(nInputs, nChannels, nFilters, kernelSize, strides, n, input, weights, bias):
    Input = [[str(input[i][j] % p) for j in range(nChannels)] for i in range(nInputs)]
    Weights = [[[str(weights[i][j][k] % p) for k in range(nFilters)] for j in range(nChannels)] for i in range(kernelSize)]
    Bias = [str(bias[i] % p) for i in range(nFilters)]
    out = [[0 for _ in range(nFilters)] for j in range((nInputs - kernelSize)//strides + 1)]
    remainder = [[None for _ in range(nFilters)] for _ in range((nInputs - kernelSize)//strides + 1)]
    for i in range((nInputs - kernelSize)//strides + 1):
        for j in range(nFilters):
            for k in range(kernelSize):
                for l in range(nChannels):
                    out[i][j] += input[i*strides + k][l]*weights[k][l][j]
            out[i][j] += bias[j]
            remainder[i][j] = str(out[i][j] % n)
            out[i][j] = str(out[i][j] // n % p)
    return Input, Weights, Bias, out, remainder

def Conv2DInt(nRows, nCols, nChannels, nFilters, kernelSize, strides, n, input, weights, bias):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    Weights = [[[[str(weights[i][j][k][l] % p) for l in range(nFilters)] for k in range(nChannels)] for j in range(kernelSize)] for i in range(kernelSize)]
    Bias = [str(bias[i] % p) for i in range(nFilters)]
    out = [[[0 for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    remainder = [[[None for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    for i in range((nRows - kernelSize)//strides + 1):
        for j in range((nCols - kernelSize)//strides + 1):
            for m in range(nFilters):
                for k in range(nChannels):
                    for x in range(kernelSize):
                        for y in range(kernelSize):
                            out[i][j][m] += input[i*strides+x][j*strides+y][k] * weights[x][y][k][m]
                out[i][j][m] += bias[m]
                remainder[i][j][m] = str(out[i][j][m] % n)
                out[i][j][m] = str(out[i][j][m] // n % p)
    return Input, Weights, Bias, out, remainder

def DenseInt(nInputs, nOutputs, n, input, weights, bias):
    Input = [str(input[i] % p) for i in range(nInputs)]
    Weights = [[str(weights[i][j] % p) for j in range(nOutputs)] for i in range(nInputs)]
    Bias = [str(bias[i] % p) for i in range(nOutputs)]
    out = [0 for _ in range(nOutputs)]
    remainder = [None for _ in range(nOutputs)]
    for j in range(nOutputs):
        for i in range(nInputs):
            out[j] += input[i] * weights[i][j]
        out[j] += bias[j]
        remainder[j] = str(out[j] % n)
        out[j] = str(out[j] // n % p)
    return Input, Weights, Bias, out, remainder

def GlobalAveragePooling2DInt(nRows, nCols, nChannels, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [0 for _ in range(nChannels)]
    remainder = [None for _ in range(nChannels)]
    for k in range(nChannels):
        for i in range(nRows):
            for j in range(nCols):
                out[k] += input[i][j][k]
        remainder[k] = str(out[k] % (nRows * nCols))
        out[k] = str(out[k] // (nRows * nCols) % p)
    return Input, out, remainder

def GlobalMaxPooling2DInt(nRows, nCols, nChannels, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [max(input[i][j][k] for i in range(nRows) for j in range(nCols)) for k in range(nChannels)]
    return Input, out

def MaxPooling2DInt(nRows, nCols, nChannels, poolSize, strides, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [[[str(max(input[i*strides + x][j*strides + y][k] for x in range(poolSize) for y in range(poolSize)) % p) for k in range(nChannels)] for j in range((nCols - poolSize) // strides + 1)] for i in range((nRows - poolSize) // strides + 1)]
    return Input, out

def Flatten2DInt(nRows, nCols, nChannels, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [Input[i][j][k] for i in range(nRows) for j in range(nCols) for k in range(nChannels)]
    return Input, out
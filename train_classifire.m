imds = imageDatastore('Flowers',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

figure
numImages = length(imds.Files);
perm = randperm(numImages,25);
for i = 1:25
    subplot(5,5,i);
    imshow(imds.Files{perm(i)});
    drawnow
end

augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',[-180 180],...
    'RandXScale',[1 4], ...
    'RandYReflection',true, ...
    'RandYScale',[1 4])

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomize');

imageSize = [64 64 3];
datastore = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter)

layers = [ ...
    imageInputLayer(imageSize,'Name','input')  
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    fullyConnectedLayer(32)
    reluLayer   
    fullyConnectedLayer(16)
    reluLayer   
    fullyConnectedLayer(8)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer ];

% lgraph = layerGraph(layers);
% figure
% plot(lgraph)

options = trainingOptions('sgdm', ...
    'MaxEpochs',100,...
    'InitialLearnRate',1e-4, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(datastore,layers,options);
analyzeNetwork(net)
%numel(net.Layers(end).ClassNames)

imdsTest_rsz = augmentedImageDatastore(imageSize,imdsTest,'DataAugmentation',augmenter)
YPred = classify(net,imdsTest_rsz);
%YTest = imdsTest.Labels;
%accuracy = sum(YPred == YTest)/numel(YTest)

figure
idx = randperm(length(imdsTest_rsz.Files),25);
for i = 1:25
    subplot(5,5,i);
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

save net

I = imread(".\Flowers\daisy\image_0854.jpg");
I2= imresize(I,[64,64],'nearest');
[Pred,scores] = classify(net,I2);
scores = max(double(scores*100));
imshow(I);
title(join([string(Pred),'' ,scores ,'%']))

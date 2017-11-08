%% Load pictures
%ship = imread('ship.png');
%iceburg = imread('8ZKRcp4.png');

loadPics = importdata('../data/train.txt');
jsonDecoded = jsondecode(char(loadPics(1)));
clear loadPics;
%% Show an image
clc
n = 100;
Img = reshape(jsonDecoded(n).band_1,[75,75]);
Img = mat2gray(Img);

jsonDecoded(n).is_iceberg


epsilon = 0.1;
numberimfs = 6;
conn = '8m';


[ix,resx,medel]=IEMD_public(Img,epsilon,numberimfs,conn);
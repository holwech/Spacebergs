% Plot
figure(1);
close all
tempImg = Img;
for i = 1:numberimfs
   subplot(1,numberimfs,i),imshow(ix(:,:,i))
   tempImg = tempImg+ix(:,:,i);
end
figure(2);
subplot(1, 2, 1), imshow(ix(:,:,1))
subplot(1, 2, 2), imshow(tempImg)
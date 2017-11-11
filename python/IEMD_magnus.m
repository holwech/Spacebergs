function  [tempImg, ix]=IEMD_magnus(image)
    %% Show an image
    n = 200;
    Img = image;
    minimal = -min(min(Img));
    Img = imadjust(Img./minimal+1,[0 1],[]);

    % Run - http://aquador.vovve.net/IEMD/index.html
    epsilon = 0.1;
    numberimfs = 6;
    conn = '8m';

    [ix,resx,medel]=IEMD_public(Img,epsilon,numberimfs,conn);
    % Plot
    tempImg = Img;
    for i = 1:numberimfs
       tempImg = tempImg+ix(:,:,i);
       figure
       imshow(tempImg)
    end
    
end
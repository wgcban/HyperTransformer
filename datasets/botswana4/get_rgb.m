function xrgb = get_rgb(x)
    R=29;
    G=20;
    B=12;

    xr = x(:,:,R);
    xg = x(:,:,G);
    xb = x(:,:,B);
    xrgb = cat(3, xr, xg, xb)/max(x, [], "all");
end


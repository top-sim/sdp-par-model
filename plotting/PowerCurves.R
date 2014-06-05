library(ggplot2)
library(Cairo)


d <- read.table("maxout.gnuplot_pipes");
names(d) <- c("Bmax", "NChannels", "Power")
v <- ggplot(d, aes(Bmax/1e3, NChannels , z = Power /1e6));
v2 <- v+ geom_tile(aes(fill = Power/1e6))  +stat_contour(binwidth = 1) + scale_fill_gradient(low="yellow", high="red");


CairoPDF("test.pdf", width=7, height=7); v2 +  theme_bw();  dev.off();

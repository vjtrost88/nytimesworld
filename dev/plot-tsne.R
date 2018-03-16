#Author: Vince Trost
#Date: 3/12/2018
#Plot t-SNE from NYT images

library(data.table)
library(ggplot2)

FILEPATH = "/storage/home/vpt5014/work/NYTimesWorld/feature-maps/t-sne-out-vgg16-feature-map.csv"

main <- function(){
  dat <- fread(FILEPATH, header=FALSE, col.names = c("tsne1", "tsne2"))
  g <- ggplot(dat, aes(x = tsne1, y = tsne2)) + geom_point()
  ggsave(filename = "tsne-plot.pdf", plot = g)
}

main()

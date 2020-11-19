from django.db import models

# Create your models here.

class Image(models.Model):
	image = models.ImageField(upload_to='user_images')

	def __str__(self):
		return str(self.image.name)

# class Tag(models.Model):
# 	label = models.CharField(max_length = 200)

# 	def __str__(self):
# 		return str(self.label)

class ImageTagRelationship(models.Model):
	image = models.ForeignKey('Image', on_delete = models.CASCADE)
	tag = models.CharField(max_length = 200)

	class Meta:
	    unique_together = (('image', 'tag'),)
	# tag = models.ForeignKey('Tag', on_delete = models.CASCADE)
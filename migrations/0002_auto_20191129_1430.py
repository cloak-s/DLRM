# Generated by Django 2.0 on 2019-11-29 06:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='patient',
            options={'verbose_name': '病人信息', 'verbose_name_plural': '病人信息'},
        ),
        migrations.AlterModelOptions(
            name='usimage',
            options={'verbose_name': '超声图像', 'verbose_name_plural': '超声图像'},
        ),
        migrations.AlterField(
            model_name='usimage',
            name='result',
            field=models.CharField(choices=[('-1', '未知'), ('0', '正常'), ('1', '患病')], default='-1', max_length=5, verbose_name='筛查结果'),
        ),
    ]

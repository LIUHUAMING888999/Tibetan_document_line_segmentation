function insert_word(filename,H)
%利用MATLAB生成Word文档
% 原摘自xiezhh，根据论坛上的相关建议，做了稍微的改动和完善
 
filespec_user =filename;
 
%===启用word调用功能========================================================
try
Word = actxGetRunningServer('Word.Application');
catch
Word = actxserver('Word.Application');
end
Word.Visible = 1; % 使word为可见；或set(Word, 'Visible', 1);
%===打开word文件，如果路径下没有则创建一个空白文档打开========================
if exist(filespec_user,'file');
Document = Word.Documents.Open(filespec_user);
else
Document = Word.Documents.Add;
Document.SaveAs2(filespec_user);
end
%===格式定义===============================================================
Content = Document.Content;
Selection = Word.Selection;
Paragraphformat = Selection.ParagraphFormat;
%===文档的页边距===========================================================
Document.PageSetup.TopMargin = 20;
Document.PageSetup.BottomMargin = 45;
Document.PageSetup.LeftMargin = 45;
Document.PageSetup.RightMargin = 20;
%==========================================================================
 
%===文档组成部分============================================================
% 文档的标题及格式
headline = '报告';
Content.Start = 0; % 起始点为0，即表示每次写入覆盖之前资料
Content.Text = headline;
Content.Font.Size = 16; % 字体大小
Content.Font.Bold = 1; % 字体加粗
Content.Paragraphs.Alignment = 'wdAlignParagraphCenter'; % 居中,wdAlignParagraphLeft/Center/Right
 
% 文档的创建时间
Selection.Start = Content.end; % 开始的地方在上一个的结尾
Selection.TypeParagraph; % 插入一个新的空段落
% 
% for i =1: length(ls('reverse_similar_patch\'))-2
% I_On= imread(['reverse_similar_patch\', num2str(i) '.png']);

% H=figure();
% imhist(I_On);
print(H,'-dbitmap');%将图片发到剪切板
Selection=Word.Selection;   
Selection.Range.Paste;%在当前光标的位置插入图片

 
  
 
Document.ActiveWindow.ActivePane.View.Type = 'wdPrintView';
Document.Save; % 保存文档
Word.Quit; % 关闭文档
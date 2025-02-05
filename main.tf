provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "source_code_bucket" {
  bucket = "my-source-code-bucket-${random_id.bucket_id.hex}"
}

resource "aws_ecr_repository" "my_repository" {
  name         = "my-ecr-repo"
  force_delete = true
}

resource "aws_iam_role" "codebuild_role" {
  name = "codebuild-role-${random_id.role_id.hex}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "codebuild_policy" {
  role = aws_iam_role.codebuild_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:CompleteLayerUpload",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:DescribeRepositories"
        ]
        Resource = "*"
      }
    ]
  })
}

data "archive_file" "source_code_zip" {
  type        = "zip"
  source_dir  = "${path.module}/source"
  output_path = "${path.module}/source.zip"
}

resource "aws_s3_object" "source_code_zip" {
  bucket = aws_s3_bucket.source_code_bucket.bucket
  key    = "source.zip"
  source = data.archive_file.source_code_zip.output_path
}

resource "aws_codebuild_project" "my_codebuild_project" {
  name          = "my-codebuild-project"
  service_role  = aws_iam_role.codebuild_role.arn
  source {
    type            = "S3"
    location        = "${aws_s3_bucket.source_code_bucket.bucket}/source.zip"
    buildspec       = file("buildspec.yml")
  }
  artifacts {
    type            = "NO_ARTIFACTS"
  }
  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:4.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true
  }
}

resource "random_id" "bucket_id" {
  byte_length = 4
}

resource "random_id" "role_id" {
  byte_length = 4
}

resource "null_resource" "trigger_codebuild" {
  provisioner "local-exec" {
    command = "ping 127.0.0.1 -n 31 > nul"
  }

  provisioner "local-exec" {
    command = "aws codebuild start-build --region ${var.region} --project-name ${aws_codebuild_project.my_codebuild_project.name}"
  }

  depends_on = [
    aws_s3_bucket.source_code_bucket,
    aws_ecr_repository.my_repository,
    aws_codebuild_project.my_codebuild_project,
    aws_s3_object.source_code_zip
  ]
}

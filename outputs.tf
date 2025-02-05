output "s3_bucket_name" {
  value = aws_s3_bucket.source_code_bucket.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.my_repository.repository_url
}

output "codebuild_project_name" {
  value = aws_codebuild_project.my_codebuild_project.name
}
